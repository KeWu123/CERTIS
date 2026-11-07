"""Independent post-execution ledger, count-coverage and full-affine audit.

Hidden truth is read here, after every policy has finished, directly from
the pinned original parquet through DuckDB. No verification label is passed
back to a policy. This check augments the actual SQL materializations saved
by the runner; it does not substitute a count for that execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import gzip
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_cases


def labels_sql(case):
    con = duckdb.connect()
    frame = pd.DataFrame({"id": np.arange(case.n), "row_id": case.row_ids,
                          "predicate_id": [case.predicate_left if s == 0 else case.predicate_right for s in case.side]})
    con.register("identities", frame)
    rows = con.execute("SELECT i.id,o.label FROM identities i JOIN read_parquet(?) o ON i.row_id=o.row_id AND i.predicate_id=o.predicate_id ORDER BY i.id",
                       [case.metadata["oracle_parquet"]]).fetchall()
    con.close()
    if [int(j) for j, y in rows] != list(range(case.n)):
        raise AssertionError("Pinned original SQL tape does not have one label per identity")
    return [int(y) for j, y in rows]


def check_record(record, case, truth):
    if record["status"] != "pass":
        return {"key": record["key"], "status": "runtime_failure", "error": record.get("error")}
    r = record["result"]; summary = r["summary"]
    ledger = r.get("oracle_ledger", r.get("ledger"))
    facts = {int(j): y for j, y in r["facts"].items()}
    assert len(ledger) == len({a["id"] for a in ledger}) == summary["oracle_requests"]
    assert facts == {a["id"]: a["label"] for a in ledger}
    assert all(truth[j] == y for j, y in facts.items())
    assert sum(a["tokens_in"] for a in ledger) == summary["oracle_tokens_in"]
    assert sum(a["tokens_out"] for a in ledger) == summary["oracle_tokens_out"]
    assert dict(Counter(a["phase"] for a in ledger)) == {k: v for k, v in summary["phase_counts"].items() if v}
    helper = r["helper_ledger"]
    assert len(helper) == len({a["id"] for a in helper}) == summary["helper_requests"]
    assert sum(a["tokens_in"] for a in helper) == summary["helper_tokens_in"]
    assert sum(a["tokens_out"] for a in helper) == summary["helper_tokens_out"]
    assert record["target_version"] == case.target_version
    assert record["oracle_version"] == case.oracle_version
    assert record["metadata"]["source_hashes"] == case.metadata["source_hashes"]
    out = r["output"]
    full = sum(sum(truth[j] for j in left)*sum(truth[j] for j in right)
               for left, right in zip(case.key_left, case.key_right))
    returned = sum(sum(facts.get(j, 0) for j in left)*sum(facts.get(j, 0) for j in right)
                   for left, right in zip(case.key_left, case.key_right))
    assert (full, returned) == (out["full_count"], out["returned_count"])
    assert out["bag_subset"] and full - returned == out["lost_count"]
    assert out["materialized_table"] == "returned_derivations"
    assert "CREATE OR REPLACE TEMP TABLE returned_derivations" in out["returned_sql"]
    affine = None
    if record["method"].startswith("cut_"):
        weights = {int(j): int(w) for j, w in r["affine_weights"].items()}
        cut_facts = {int(j): y for j, y in r["cut_facts"].items()}
        assert not weights.keys() & cut_facts.keys()
        assert set(weights) | set(cut_facts) == set(range(case.n))
        affine = int(r["affine_constant"]) + sum(w*truth[j] for j, w in weights.items())
        assert affine == full, "FULL affine identity fails; returned-only checks cannot detect this"
    elif record["method"] == "cover":
        d = r["diagnostics"]
        affine = d["residual_constant"] + sum(int(w)*truth[int(j)] for j, w in d["residual_weights"])
        assert affine == full
    regions = r.get("regions", [])
    covered = all(a["lower"] <= sum(truth[j] for j in a["ids"]) <= a["upper"] for a in regions)
    alpha = sum((Fraction(a["alpha_executed"]) for a in regions if a.get("alpha_executed") is not None), Fraction(0))
    assert alpha <= Fraction(1, 20)
    if covered: assert 10*(full-returned) <= full
    discarded = set(r.get("discarded", ()))
    assert not discarded & facts.keys()
    if "initial_retained_unknown" in r:
        retained = set(r["initial_retained_unknown"])
        assert {a["id"] for a in ledger if a["phase"] == "execution"} <= retained
    return {"key": record["key"], "status": "pass", "joint_count_coverage_observed": covered,
            "observed_loss_target_met": 10*(full-returned) <= full,
            "full_affine_count": affine, "full_affine_verified": affine is not None,
            "executed_alpha_sum": str(alpha), "full_count": full, "returned_count": returned}


def validate(directory, *, expected=None):
    paths = sorted((directory / "raw").glob("*.json.gz"))
    cases = {c.case_id: c for c in load_cases()}
    truths = {}; checks = []; failures = []; records = {}
    for path in paths:
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["key"] in records:
            failures.append({"key": record["key"], "error": "Duplicate/retry requires explicit resolution"})
            continue
        # Keep only the identity sets needed for paired containment. Holding
        # every parsed raw ledger/region/network would scale to gigabytes.
        r = record.get("result", {})
        records[record["key"]] = {"status": record["status"], "method": record["method"],
            "initial_mask_index": r.get("initial_mask_index"),
            "initial_retained_unknown": r.get("initial_retained_unknown"),
            "paid_ids": {a["id"] for a in r.get("oracle_ledger", r.get("ledger", []))}}
        case = cases[record["case_id"]]
        if case.case_id not in truths: truths[case.case_id] = labels_sql(case)
        try:
            check = check_record(record, case, truths[case.case_id]); checks.append(check)
            if check["status"] != "pass": failures.append(check)
        except Exception as exc:
            failures.append({"key": record["key"], "error": repr(exc)})
    # Compare only same-source/same-start static and feedback references.
    containment_pairs = 0
    for key, record in records.items():
        if record["status"] != "pass": continue
        method = record["method"]
        if method in ("direct_feedback_audit_helper", "cut_feedback_audit"):
            static_key = key.replace("feedback", "static")
            if static_key in records and records[static_key]["status"] == "pass":
                a, b = record, records[static_key]
                assert a["initial_mask_index"] == b["initial_mask_index"]
                assert a["initial_retained_unknown"] == b["initial_retained_unknown"]
                assert a["paid_ids"] <= b["paid_ids"]
                containment_pairs += 1
    if expected is not None and len(records) != expected:
        failures.append({"error": f"Expected {expected} unique records, found {len(records)}"})
    result = {"status": "pass" if not failures else "failure", "records": len(records),
              "checked_cases": len(truths), "containment_pairs": containment_pairs,
              "full_affine_identity_checks": sum(a.get("full_affine_verified", False) for a in checks),
              "observed_coverage_failures": [a["key"] for a in checks if not a.get("joint_count_coverage_observed", True)],
              "observed_loss_violations": [a["key"] for a in checks if not a.get("observed_loss_target_met", True)],
              "failures": failures, "checks": checks,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Original-parquet post-execution verification; labels never returned to a policy"}
    (directory / "post_execution_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("directory", type=Path); p.add_argument("--expected", type=int)
    args = p.parse_args(); result = validate(args.directory, expected=args.expected)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
    if result["status"] != "pass": raise SystemExit(1)
