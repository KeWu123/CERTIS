"""Exactly 288 exploratory post-matrix mechanism runs with frozen bindings."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import traceback

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from common import INPUTS, SOURCE, load_cases, potential_degrees
from mechanism_controls import METHODS, REGISTRATION_NOTE, run_mechanism
from validate_results import check_record, labels_sql

SID = "db17c4543b8832d2acf2fc3db26099461a7055dc1ed64ad0867c0e5a47c8c22f"
SEEDS = (101, 202, 303)
BUDGET = 500
ORIGINAL = "direct_feedback_audit_degree"
EXPECTED = 288


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def read_raw(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save_raw(path, value):
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    atomic_bytes(path, gzip.compress(body, compresslevel=5, mtime=0))


def log(output, value):
    line = json.dumps({"utc": now(), **value}, ensure_ascii=False)
    with (output / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def key_for(case, method, seed):
    return f"{case.case_id}__{method}__s{seed}__b{BUDGET}"


def baseline_path(primary, case, seed):
    return primary / "raw" / (key_for(case, ORIGINAL, seed) + ".json.gz")


def freeze(primary, output):
    original_manifest_path = HERE / "results/sources" / SID / "manifest.json"
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() == SID
    completion = json.loads((primary / "full_completion.json").read_text(encoding="utf-8"))
    validation = json.loads((primary / "post_execution_validation.json").read_text(encoding="utf-8"))
    assert completion["executed_fail"] == 0 and completion["expected_unique_runs"] == 1376
    assert validation["status"] == "pass" and validation["records"] == 1376
    shared = [HERE / "common.py", HERE / "direct_arms.py",
              ROOT / "code/bilinear_certificate.py", ROOT / "code/exact_hypergeom.py"]
    for path in shared:
        assert digest(path) == original[relative(path)], f"Original shared source changed: {path}"
    paths = [HERE / name for name in ("mechanism_controls.py", "mechanism_controls_checks.py",
             "mechanism_controls_checks.json", "run_mechanism_study.py", "MECHANISM_PROTOCOL.md",
             "common.py", "direct_arms.py", "validate_results.py")]
    paths += [ROOT / "code/bilinear_certificate.py", ROOT / "code/exact_hypergeom.py",
              HERE.parent / "EXTERNAL_PROTOCOL.md", INPUTS / "validation.json",
              SOURCE / "download_manifest.json", original_manifest_path,
              primary / "full_completion.json", primary / "post_execution_validation.json"]
    inputs = [INPUTS / f"{dataset}_arrays.npz" for dataset in ("dbpedia", "goemotions")]
    inputs += [SOURCE / f"data/datasets/{dataset}.parquet" for dataset in ("dbpedia", "goemotions")]
    inputs += [SOURCE / f"data/tapes/{dataset}__oracle-llama.parquet" for dataset in ("dbpedia", "goemotions")]
    comparisons = {}
    for seed in SEEDS:
        matches = sorted((primary / "raw").glob(f"*__{ORIGINAL}__s{seed}__b{BUDGET}.json.gz"))
        assert len(matches) == 32
        comparisons.update({relative(path): digest(path) for path in matches})
    assert len(comparisons) == 96
    binding = {"primary_source_id": SID, "protocol_date": "2026-10-16 Asia/Shanghai",
               "sources": {relative(path): digest(path) for path in paths},
               "pinned_inputs": {relative(path): digest(path) for path in inputs},
               "matched_original_records": comparisons, "methods": list(METHODS),
               "seeds": list(SEEDS), "budget_per_side": BUDGET, "expected_runs": EXPECTED,
               "post_matrix_control": True, "exploratory": True, "preregistered": False,
               "registration_note": REGISTRATION_NOTE}
    source_id = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {**binding, "supplemental_source_id": source_id}
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest, "Refuse mixed-source resumption"
    else:
        atomic_json(manifest_path, manifest)
    frozen = output / "sources" / source_id
    for path in paths:
        target = frozen / relative(path)
        if not target.exists():
            atomic_bytes(target, path.read_bytes())
        assert digest(target) == manifest["sources"][relative(path)]
    atomic_json(frozen / "manifest.json", manifest)
    verify_sources(output, manifest)
    return manifest, digest(manifest_path)


def verify_sources(output, manifest, *, full=True):
    frozen = output / "sources" / manifest["supplemental_source_id"]
    for name, expected in manifest["sources"].items():
        assert digest(ROOT / name) == digest(frozen / name) == expected, f"Source bytes changed: {name}"
    if full:
        for group in ("pinned_inputs", "matched_original_records"):
            for name, expected in manifest[group].items():
                assert digest(ROOT / name) == expected, f"Bound input changed: {name}"
    assert json.loads((output / "source_manifest.json").read_text(encoding="utf-8")) == manifest


def audit_rows(result):
    return [row for row in result["oracle_ledger"] if row["phase"] == "audit"]


def validate(record, case, truth, original, original_path, order, manifest, manifest_hash):
    assert record["source_id"] == manifest["supplemental_source_id"]
    assert record["source_manifest_sha256"] == manifest_hash and record["primary_source_id"] == SID
    assert record["matched_original_sha256"] == digest(original_path) == manifest["matched_original_records"][relative(original_path)]
    assert record["seed"] in SEEDS and record["budget"] == BUDGET and record["method"] in METHODS
    assert record["post_matrix_control"] and record["exploratory"] and not record["preregistered"]
    checked = check_record(record, case, truth)
    assert checked["status"] == "pass", checked
    assert original["status"] == "pass" and original["source_id"] == SID
    assert original["target_version"] == case.target_version and original["oracle_version"] == case.oracle_version
    result, base = record["result"], original["result"]
    assert result["regions"] == base["regions"], "Audit region records differ from matched original"
    assert audit_rows(result) == audit_rows(base), "Ordered paid audit ledger differs from matched original"
    assert result["summary"]["helper_requests"] == result["summary"]["helper_read_calls"] == result["summary"]["helper_read_items"] == 0
    assert result["helper_ledger"] == [] and result["summary"]["initial_cache_count"] == 0
    assert result["mask_family_size"] == 65 and result["execution_batch_size"] == 128
    assert result["dynamic_partner_pruning"]
    audit_facts = {row["id"]: row["label"] for row in audit_rows(result)}
    assert {int(j): value for j, value in result["initial_facts_after_audit"].items()} == audit_facts
    initial_card = case.n * result["initial_mask_index"] // 64
    prefix = set(int(j) for j in order[:initial_card])
    r0 = sorted(set(range(case.n)) - prefix - audit_facts.keys())
    effective_initial = len(prefix - audit_facts.keys())
    assert result["initial_R0"] == result["initial_retained_unknown"] == r0
    assert result["initial_certificate"]["discarded_unknown_count"] == effective_initial
    assert result["initial_certificate"]["fact_count"] == len(audit_facts)
    assert result["final_certificate"]["discarded_unknown_count"] == len(result["discarded"])
    assert result["final_certificate"]["fact_count"] == result["summary"]["oracle_requests"]
    assert result["final_certificate"]["possible_retained_unknown_count"] == 0
    indices = [event["mask_index"] for event in result["events"]]
    assert indices[0] == result["initial_mask_index"] and indices[-1] == result["final_mask_index"]
    assert indices == sorted(indices)
    for event in result["events"]:
        assert event["discard_cardinality"] == case.n * event["mask_index"] // 64
    conflict_events = [index for index, event in enumerate(result["events"]) if event["conflict"]]
    if conflict_events:
        assert len(set(indices[conflict_events[0]:])) == 1
        assert result["conflict_freeze"]
    if record["method"] == "fixed_mask_pruning_degree":
        assert result["initial_mask_index"] == result["final_mask_index"] == base["initial_mask_index"]
        assert result["initial_retained_unknown"] == base["initial_retained_unknown"]
        assert len(set(indices)) == 1
        assert set(result["discarded"]) == prefix - audit_facts.keys()
    checked.update(ordered_audit_ledger_identical=True, audit_regions_identical=True,
                   fixed_mask_verified=record["method"] == "fixed_mask_pruning_degree")
    return checked


def metrics(record):
    result = record["result"]
    summary, out = result["summary"], result["output"]
    audited = len(audit_rows(result))
    return {"requests": summary["oracle_requests"], "oracle_tokens_in": summary["oracle_tokens_in"],
            "oracle_tokens_out": summary["oracle_tokens_out"], "helper_requests": summary["helper_requests"],
            "helper_tokens_in": summary["helper_tokens_in"], "helper_tokens_out": summary["helper_tokens_out"],
            "initial_mask_index": result["initial_mask_index"], "final_mask_index": result["final_mask_index"],
            "initial_base_discard_cardinality": record["n"] * result["initial_mask_index"] // 64,
            "final_base_discard_cardinality": record["n"] * result["final_mask_index"] // 64,
            "initial_effective_unknown_discarded": record["n"] - audited - result["initial_retained_unknown_count"],
            "final_effective_unknown_discarded": len(result["discarded"]),
            "initial_retained_unknown": result["initial_retained_unknown_count"],
            "full_count": out["full_count"], "returned_count": out["returned_count"],
            "lost_count": out["lost_count"], "loss": out["loss"], "conflict_freeze": result["conflict_freeze"],
            "algorithm_seconds": result["timings"]["algorithm_seconds"],
            "preprocessing_seconds": result["timings"]["preprocessing_seconds"],
            "audit_seconds": result["timings"]["audit_seconds"],
            "projection_seconds": result["timings"]["projection_seconds"],
            "execution_bookkeeping_seconds": result["timings"]["execution_bookkeeping_seconds"],
            "process_cpu_seconds_including_materialization": record.get("process_cpu_seconds_including_materialization"),
            "returned_materialization_seconds": out["returned_materialization_seconds"],
            "sql_validation_seconds": out["sql_validation_seconds"]}


def save_sql(output, record):
    out = record["result"]["output"]
    text = f"-- {record['key']}\n-- target_version: {record['target_version']}\n"
    text += "-- Executed by common.evaluate; facts are in the paired raw ledger.\n"
    text += out["reference_sql"] + ";\n\n" + out["returned_sql"] + ";\n"
    atomic_bytes(output / "sql" / (record["key"] + ".sql"), text.encode("utf-8"))


def verify_result_hashes(output, manifest, manifest_hash):
    saved = json.loads((output / "result_manifest.json").read_text(encoding="utf-8"))
    assert saved["source_id"] == manifest["supplemental_source_id"] and saved["source_manifest_sha256"] == manifest_hash
    for group, folder in (("raw", "raw"), ("sql", "sql")):
        assert len(saved[group]) == EXPECTED
        assert set(saved[group]) == {path.name for path in (output / folder).iterdir() if path.is_file()}
        for name, expected in saved[group].items():
            assert digest(output / folder / name) == expected, f"Saved result bytes changed: {name}"
    for name, expected in saved["derived"].items():
        assert digest(output / name) == expected, f"Saved derived bytes changed: {name}"


def summarize(primary, output, cases, manifest, manifest_hash):
    checks, failures, rows = [], [], []
    paths = sorted((output / "raw").glob("*.json.gz"))
    assert len(paths) == EXPECTED
    for case in cases:
        truth = labels_sql(case)
        order = np.lexsort((np.arange(case.n), potential_degrees(case, {})))
        for seed in SEEDS:
            source_path = baseline_path(primary, case, seed)
            original = read_raw(source_path)
            original_metrics = metrics(original)
            for method in METHODS:
                path = output / "raw" / (key_for(case, method, seed) + ".json.gz")
                record = read_raw(path)
                try:
                    checked = validate(record, case, truth, original, source_path, order, manifest, manifest_hash)
                    checked["raw_sha256"] = digest(path)
                    checks.append(checked)
                    current = metrics(record)
                    delta = current["requests"] - original_metrics["requests"]
                    row = {"case_id": case.case_id, "dataset": case.dataset, "variant": case.variant,
                           "method": method, "seed": seed, "budget": BUDGET, **current,
                           **{"original_" + name: value for name, value in original_metrics.items()},
                           "control_minus_original_requests": delta,
                           "request_comparison": "win" if delta < 0 else "tie" if delta == 0 else "loss",
                           "joint_count_coverage_observed": checked["joint_count_coverage_observed"],
                           "observed_loss_target_met": checked["observed_loss_target_met"]}
                    rows.append(row)
                except Exception as exc:
                    failures.append({"key": record["key"], "error": repr(exc)})
    validation = {"status": "pass" if not failures and len(checks) == EXPECTED else "failure",
                  "records": EXPECTED, "independently_passed": len(checks), "cases": len(cases),
                  "ordered_audit_and_region_matches": len(checks),
                  "fixed_mask_checks": sum(check["fixed_mask_verified"] for check in checks),
                  "checks": checks, "failures": failures, "source_id": manifest["supplemental_source_id"],
                  "source_manifest_sha256": manifest_hash, "post_matrix_control": True,
                  "exploratory": True, "preregistered": False, "registration_note": REGISTRATION_NOTE}
    atomic_json(output / "validation.json", validation)
    if failures:
        return {"status": "failure", "failures": failures}
    aggregates = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate = {"method": method, "runs": len(selected),
                     "control_wins": sum(row["request_comparison"] == "win" for row in selected),
                     "ties": sum(row["request_comparison"] == "tie" for row in selected),
                     "control_losses": sum(row["request_comparison"] == "loss" for row in selected),
                     "conflict_count": sum(row["conflict_freeze"] for row in selected),
                     "observed_coverage_failures": sum(not row["joint_count_coverage_observed"] for row in selected),
                     "observed_loss_violations": sum(not row["observed_loss_target_met"] for row in selected),
                     "maximum_loss": max(row["loss"] for row in selected)}
        for name in ("requests", "oracle_tokens_in", "oracle_tokens_out", "helper_requests", "helper_tokens_in", "helper_tokens_out",
                     "initial_effective_unknown_discarded", "final_effective_unknown_discarded", "algorithm_seconds"):
            aggregate[name] = sum(row[name] for row in selected)
            aggregate["original_" + name] = sum(row["original_" + name] for row in selected)
        aggregate["control_minus_original_requests"] = aggregate["requests"] - aggregate["original_requests"]
        aggregate["process_cpu_seconds_including_materialization"] = sum(row["process_cpu_seconds_including_materialization"] for row in selected)
        aggregates.append(aggregate)
    summary = {"status": "pass", "runs": EXPECTED, "cases": 32, "seeds": list(SEEDS), "budget": BUDGET,
               "aggregates": aggregates, "rows": rows, "source_id": manifest["supplemental_source_id"],
               "source_manifest_sha256": manifest_hash, "post_matrix_control": True, "exploratory": True,
               "preregistered": False, "registration_note": REGISTRATION_NOTE,
               "comparison_scope": "Each new control versus the original D_degree500 cell at identical case/seed/audit; control win means fewer requests",
               "timing_scope": "Sequential later-run component times; descriptive, not randomized speed comparisons",
               "evidence_scope": "Exploratory post-matrix dependent fixed targets; not held-out confirmation"}
    atomic_json(output / "summary.json", summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(output / "per_cell.csv", stream.getvalue().encode("utf-8-sig"))
    atomic_json(output / "result_manifest.json", {
        "source_id": manifest["supplemental_source_id"], "source_manifest_sha256": manifest_hash,
        "raw": {path.name: digest(path) for path in paths},
        "sql": {path.name: digest(path) for path in sorted((output / "sql").glob("*.sql"))},
        "derived": {name: digest(output / name) for name in ("validation.json", "summary.json", "per_cell.csv")},
        "registration_note": REGISTRATION_NOTE})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    primary = HERE / "results" / SID
    output = primary / "mechanism_controls"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": now(), "original_argv": sys.orig_argv, "executable": sys.executable,
                                 "cwd": str(Path.cwd()), "verify_only": args.verify_only}) + "\n")
    manifest, manifest_hash = freeze(primary, output)
    machine = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "duckdb": duckdb.__version__, "processor": platform.processor(),
               "sql_threads": 1, "live_inference": False, "network_calls": False}
    atomic_json(output / "environment.json", machine)
    cases = load_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 32
    if (output / "result_manifest.json").exists():
        verify_result_hashes(output, manifest, manifest_hash)
    executed = reused = failures = 0
    if not args.verify_only:
        for case in cases:
            truth = None
            order = np.lexsort((np.arange(case.n), potential_degrees(case, {})))
            for seed in SEEDS:
                source_path = baseline_path(primary, case, seed)
                original = read_raw(source_path)
                for method in METHODS:
                    verify_sources(output, manifest, full=False)
                    key = key_for(case, method, seed)
                    path = output / "raw" / (key + ".json.gz")
                    if path.exists() and read_raw(path)["status"] == "pass":
                        if truth is None:
                            truth = labels_sql(case)
                        validate(read_raw(path), case, truth, original, source_path, order, manifest, manifest_hash)
                        reused += 1
                        log(output, {"key": key, "status": "reused_verified"})
                        continue
                    started = time.perf_counter()
                    record = {"key": key, "case_id": case.case_id, "dataset": case.dataset, "variant": case.variant,
                              "predicate_left": case.predicate_left, "predicate_right": case.predicate_right,
                              "method": method, "seed": seed, "budget": BUDGET, "n": case.n,
                              "key_count": len(case.key_names), "metadata": case.metadata,
                              "target_version": case.target_version, "oracle_version": case.oracle_version,
                              "source_id": manifest["supplemental_source_id"], "primary_source_id": SID,
                              "source_manifest_sha256": manifest_hash, "matched_original_sha256": digest(source_path),
                              "matched_original_path": relative(source_path), "machine": machine,
                              "attempt_utc": now(), "attempt_id": time.time_ns(),
                              "post_matrix_control": True, "exploratory": True, "preregistered": False,
                              "registration_note": REGISTRATION_NOTE}
                    try:
                        cpu_started = time.process_time()
                        result = run_mechanism(case, method, seed, BUDGET, materialize=True)
                        record["process_cpu_seconds_including_materialization"] = time.process_time() - cpu_started
                        record.update(status="pass", result=result)
                        verification_started = time.perf_counter()
                        if truth is None:
                            truth = labels_sql(case)
                        record["independent_validation"] = validate(record, case, truth, original, source_path, order, manifest, manifest_hash)
                        record["independent_validation_seconds"] = time.perf_counter() - verification_started
                        save_sql(output, record)
                        executed += 1
                    except Exception as exc:
                        record.update(status="failure", error=repr(exc), traceback=traceback.format_exc())
                        failures += 1
                    record["wall_seconds_including_verification"] = time.perf_counter() - started
                    if record["status"] == "failure":
                        save_raw(output / "failures" / f"{key}__attempt{record['attempt_id']}.json.gz", record)
                    save_raw(path, record)
                    log(output, {"key": key, "status": record["status"], "seconds": record["wall_seconds_including_verification"],
                                 "requests": record.get("result", {}).get("summary", {}).get("oracle_requests"),
                                 "completed_this_invocation": executed, "reused": reused, "failed": failures})
            case.close_sql()
    verify_sources(output, manifest)
    summary = summarize(primary, output, cases, manifest, manifest_hash)
    verify_sources(output, manifest)
    if summary["status"] == "pass":
        verify_result_hashes(output, manifest, manifest_hash)
    completion = {"status": summary["status"], "expected_unique_runs": EXPECTED, "executed_pass": executed,
                  "executed_fail": failures, "reused_verified_runs": reused, "verify_only": args.verify_only,
                  "source_id": manifest["supplemental_source_id"], "source_manifest_sha256": manifest_hash,
                  "source_and_result_hashes_verified": summary["status"] == "pass", "utc": now(),
                  "preserved_failure_attempts": len(list((output / "failures").glob("*.json.gz"))),
                  "post_matrix_control": True, "exploratory": True, "preregistered": False,
                  "registration_note": REGISTRATION_NOTE}
    name = "verification_completion.json" if args.verify_only else "resume_completion.json" if reused == EXPECTED else "completion.json"
    atomic_json(output / name, completion)
    log(output, {"completion": completion, "aggregates": summary.get("aggregates", [])})
    if failures or summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
