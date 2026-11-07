"""AUTHOR addition: strict post-execution tape and cell validation.

This module is for a runner's verification phase only. Never pass its
PinnedReference (which contains labels and charges) to a policy. Historical
methods, runners and raw records are deliberately not modified.

Typical use AFTER the first policy finishes:
    reference = load_pinned_reference(case)
    check = strict_check_record(record, case, record_path=path,
        expected_key=key, expected_method=method, expected_seed=seed,
        expected_budget=budget, expected_source_id=source_id,
        expected_supplement_source_id=supplement_id, reference=reference)

Explicit result_method_aliases allow wrapper method names. A top-level
seed0/budget0 point control whose original result records seed101 must
declare seed0_control_seed=101. Exact no_audit controls may omit result.seed
only for the declared 0/0 cell. None for expected_supplement_source_id means
that field must be absent (original records), not a wildcard.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from common import WORKSPACE as REPOSITORY_ROOT
from typing import Mapping
import hashlib
import math

import duckdb
import numpy as np
import pandas as pd

from validate_results import check_record
from exact_hypergeom import count_interval_exact

PHASES = ("audit", "cut", "execution")
ORIGINAL_METHOD_ALIASES = {
    "scan": "exact_scan", "full_scan": "exact_scan", "left": "left_first",
    "right": "right_first", "mincut": "cover", "greedy": "helper_greedy",
    "max_degree": "dynamic_max_degree", "metadata_max_degree": "dynamic_max_degree",
    "degree_greedy": "dynamic_max_degree", "negative_first": "helper_negative_first",
    "negative_first_greedy": "helper_negative_first",
}


class StrictValidationError(AssertionError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _require(condition, code, detail):
    if not condition:
        raise StrictValidationError(code, detail)


def _integer(value, code, *, minimum=0):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)), code, "integer required")
    value = int(value)
    _require(value >= minimum, code, "integer below minimum")
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case_pairs(case):
    _require(len(case.row_ids) == case.n and len(case.ids) == case.n and len(case.side) == case.n,
             "reference.identities", "one row ID, identity and side per target identity required")
    pairs = tuple((str(row), case.predicate_left if int(side) == 0 else case.predicate_right)
                  for row, side in zip(case.row_ids, case.side))
    _require(all(int(s) in (0, 1) for s in case.side), "reference.identities", "invalid side")
    _require(len(set(case.ids)) == case.n and len(set(pairs)) == case.n,
             "reference.identities", "duplicate identity or duplicate (row_id,predicate_id)")
    return pairs


@dataclass(frozen=True)
class PinnedReference:
    case_id: str
    target_version: str
    oracle_version: str
    identities: tuple[str, ...]
    row_predicates: tuple[tuple[str, str], ...]
    labels: tuple[int, ...]
    oracle_tokens_in: tuple[int, ...]
    oracle_tokens_out: tuple[int, ...]
    helper_tokens_in: tuple[int, ...]
    helper_tokens_out: tuple[int, ...]
    helper_labels: tuple[int, ...]
    helper_scores: tuple[float, ...]
    source_hashes: tuple[tuple[str, str], ...]


def _pinned_hash(case, path):
    path = Path(path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    hashes = case.metadata.get("source_hashes", {})
    normalized = str(path).replace("\\", "/")
    matches = [(name, value) for name, value in hashes.items()
               if normalized.endswith("/" + str(name).replace("\\", "/"))]
    if not matches:
        matches = [(name, value) for name, value in hashes.items() if Path(name).name == path.name]
    _require(len(matches) == 1, "reference.pin", f"exactly one pinned hash required for {path.name}")
    name, expected = matches[0]
    _require(path.is_file() and _digest(path) == expected, "reference.hash", f"pinned bytes differ: {name}")
    return path, str(name), str(expected)


def load_pinned_reference(case) -> PinnedReference:
    """Join original oracle AND helper Parquet by BOTH row and predicate.

    No fixed target size and no row-ID-only dictionary is used. Native
    9996-identity targets may reuse each released row under two predicates.
    Missing, duplicate, null, or invalid pinned tape rows are rejected.
    """
    pairs = _case_pairs(case)
    oracle_path = case.metadata.get("oracle_parquet")
    _require(oracle_path is not None, "reference.pin", "oracle_parquet metadata required")
    helper_path = case.metadata.get("helper_parquet")
    if helper_path is None:
        name = Path(oracle_path).name
        _require(name.endswith("__oracle-llama.parquet"), "reference.pin", "explicit helper_parquet required")
        helper_path = Path(oracle_path).with_name(name.replace("__oracle-llama.parquet", "__helper.parquet"))
    oracle_path, oname, ohash = _pinned_hash(case, oracle_path)
    helper_path, hname, hhash = _pinned_hash(case, helper_path)
    con = duckdb.connect()
    try:
        con.register("strict_identities", pd.DataFrame({
            "id": np.arange(case.n, dtype=np.int64),
            "row_id": pd.Series([p[0] for p in pairs], dtype="string"),
            "predicate_id": pd.Series([p[1] for p in pairs], dtype="string")}))
        rows = con.execute("""SELECT i.id,o.row_id,h.row_id,o.label,o.tokens_in,o.tokens_out,
 h.tokens_in,h.tokens_out,h.label,h.score
FROM strict_identities i
LEFT JOIN read_parquet(?) o ON i.row_id=o.row_id AND i.predicate_id=o.predicate_id
LEFT JOIN read_parquet(?) h ON i.row_id=h.row_id AND i.predicate_id=h.predicate_id
ORDER BY i.id""", [str(oracle_path), str(helper_path)]).fetchall()
    finally:
        con.close()
    _require(len(rows) == case.n and [r[0] for r in rows] == list(range(case.n)),
             "reference.cardinality", "tape join is not one row per identity")
    labels, oi, oo, hi, ho, hl, hs = [], [], [], [], [], [], []
    for j, orow, hrow, label, tin, tout, hin, hout, helper_label, helper_score in rows:
        _require(orow is not None, "reference.oracle_missing", f"no oracle row for identity {j}")
        _require(hrow is not None, "reference.helper_missing", f"no helper row for identity {j}")
        label = _integer(label, "reference.oracle_label")
        helper_label = _integer(helper_label, "reference.helper_label")
        _require(label in (0, 1) and helper_label in (0, 1), "reference.label", "binary tape labels required")
        labels.append(label); hl.append(helper_label)
        _require(isinstance(helper_score, (int, float)) and math.isfinite(helper_score) and 0 <= helper_score <= 1,
                 "reference.helper_score", "finite helper score in [0,1] required")
        hs.append(float(helper_score))
        oi.append(_integer(tin, "reference.oracle_tokens")); oo.append(_integer(tout, "reference.oracle_tokens"))
        hi.append(_integer(hin, "reference.helper_tokens")); ho.append(_integer(hout, "reference.helper_tokens"))
    # Detect source changes during the query as well as before it.
    _require(_digest(oracle_path) == ohash and _digest(helper_path) == hhash,
             "reference.hash", "pinned tape changed during validation")
    return PinnedReference(case.case_id, case.target_version, case.oracle_version, tuple(case.ids), pairs,
                           tuple(labels), tuple(oi), tuple(oo), tuple(hi), tuple(ho), tuple(hl), tuple(hs),
                           ((oname, ohash), (hname, hhash)))


def _validate_ledger(ledger, case, reference, *, helper=False):
    prefix = "ledger.helper" if helper else "ledger.oracle"
    _require(isinstance(ledger, list), prefix, "ledger list required")
    seen = set()
    for row in ledger:
        _require(isinstance(row, Mapping), prefix, "ledger record required")
        j = _integer(row.get("id"), prefix + "_id")
        _require(j < case.n, prefix + "_id", "out-of-target identity")
        _require(j not in seen, prefix + "_duplicate", f"duplicate identity {j}")
        seen.add(j)
        _require(row.get("identity") == case.ids[j], prefix + "_identity", f"identity mismatch at {j}")
        if helper:
            if "label" in row:
                label = _integer(row["label"], prefix + "_label")
                _require(label == reference.helper_labels[j], prefix + "_label", f"label mismatch at {j}")
            if "phase" in row:
                _require(row["phase"] == "helper", prefix + "_phase", "helper phase must be implicit or helper")
            tin, tout = reference.helper_tokens_in[j], reference.helper_tokens_out[j]
        else:
            label = _integer(row.get("label"), prefix + "_label")
            _require(label == reference.labels[j], prefix + "_label", f"label mismatch at {j}")
            _require(row.get("phase") in PHASES, prefix + "_phase", "invalid oracle phase")
            tin, tout = reference.oracle_tokens_in[j], reference.oracle_tokens_out[j]
        charged_in = _integer(row.get("tokens_in"), prefix + "_tokens")
        charged_out = _integer(row.get("tokens_out"), prefix + "_tokens")
        _require((charged_in, charged_out) == (tin, tout), prefix + "_tokens", f"pinned charge mismatch at {j}")
    return seen


def _validate_summary(summary, ledger, helper):
    _require(isinstance(summary, Mapping), "summary", "summary mapping required")
    expected = {"oracle_requests": len(ledger), "helper_requests": len(helper),
                "oracle_tokens_in": sum(r["tokens_in"] for r in ledger),
                "oracle_tokens_out": sum(r["tokens_out"] for r in ledger),
                "helper_tokens_in": sum(r["tokens_in"] for r in helper),
                "helper_tokens_out": sum(r["tokens_out"] for r in helper)}
    for key, value in expected.items():
        _require(_integer(summary.get(key), "summary." + key) == value, "summary." + key, "aggregate mismatch")
    expected_counts = {phase: sum(r["phase"] == phase for r in ledger) for phase in PHASES}
    expected_tokens = {phase: {"tokens_in": sum(r["tokens_in"] for r in ledger if r["phase"] == phase),
                               "tokens_out": sum(r["tokens_out"] for r in ledger if r["phase"] == phase)} for phase in PHASES}
    for key, expected_value in (("phase_counts", expected_counts), ("phase_tokens", expected_tokens)):
        actual = summary.get(key)
        _require(isinstance(actual, Mapping) and set(actual) == set(PHASES), "summary." + key, "all phases required")
        if key == "phase_counts":
            checked = {p: _integer(actual[p], "summary." + key) for p in PHASES}
        else:
            checked = {}
            for p in PHASES:
                _require(isinstance(actual[p], Mapping) and set(actual[p]) == {"tokens_in", "tokens_out"},
                         "summary." + key, "both token directions required per phase")
                checked[p] = {d: _integer(actual[p][d], "summary." + key) for d in ("tokens_in", "tokens_out")}
        _require(checked == expected_value, "summary." + key, "phase aggregate mismatch")
    if "initial_cache_count" in summary:
        _require(_integer(summary["initial_cache_count"], "summary.initial_cache_count") == 0,
                 "summary.initial_cache_count", "this verifier's base SQL contract is cold-cache")
    # Repeated helper reads cannot be reconstructed from a deduplicated charge
    # ledger; validate their available consistency constraints, not invented calls.
    for key in ("helper_read_calls", "helper_read_items"):
        if key in summary:
            _integer(summary[key], "summary." + key)
    if "helper_read_items" in summary:
        _require(summary["helper_read_items"] >= len(helper), "summary.helper_read_items", "fewer reads than charged identities")
    if "helper_read_calls" in summary and summary["helper_read_calls"] == 0:
        _require(not helper and not summary.get("helper_read_items", 0), "summary.helper_read_calls", "charges without a read")


def _validate_regions(case, result, audit, *, required):
    if not required:
        _require(not audit, "evidence.audit_unexpected", "no-audit control acquired audit labels")
        return
    regions = result.get("regions")
    _require(isinstance(regions, list), "evidence.regions_missing", "certificate arm must retain regions")
    domains = tuple(dict.fromkeys(tuple(ids) for pair in zip(case.key_left, case.key_right) for ids in pair if ids))
    _require(len(regions) == len(domains), "evidence.regions_structure", "one ordered record per nonempty domain required")
    delta = Fraction(result.get("delta", "1/20"))
    _require(0 < delta <= Fraction(1, 20), "evidence.delta", "invalid allocated failure budget")
    nominal = delta / max(1, len(domains))
    alpha = float(nominal)
    if Fraction.from_float(alpha) > nominal:
        alpha = math.nextafter(alpha, 0.)
    _require(0 < alpha < 1, "evidence.alpha", "alpha not representable")
    sample = {r["id"]: r["label"] for r in audit}
    for ids, record in zip(domains, regions):
        _require(isinstance(record, Mapping) and isinstance(record.get("ids"), list), "evidence.regions_structure", "ordered domain required")
        checked_ids = tuple(_integer(j, "evidence.regions_structure") for j in record["ids"])
        _require(checked_ids == ids, "evidence.regions_structure", "ordered domain mismatch")
        n = sum(j in sample for j in ids); k = sum(sample.get(j, 0) for j in ids)
        lo, hi = count_interval_exact(len(ids), n, k, alpha)
        expected = dict(population=len(ids), sample=n, positive_sample=k, lower=lo, upper=hi,
                        alpha_nominal=str(nominal), alpha_executed=str(Fraction.from_float(alpha)), alpha_float_hex=alpha.hex())
        for key in ("population", "sample", "positive_sample", "lower", "upper"):
            _integer(record.get(key), "evidence.regions_values")
        _require(all(record.get(key) == value for key, value in expected.items()),
                 "evidence.regions_values", "record does not match paid SRS and exact inversion")


def _validate_cut_regions(case, result, audit, cut, reference, *, audited, budget, side_draws):
    """The paid-cut arm audits post-cut unknowns in fixed full-side bins."""
    cut_facts = {r["id"]: r["label"] for r in cut}
    raw = result.get("cut_facts", {})
    _require({int(j): y for j, y in raw.items()} == cut_facts, "evidence.cut_facts", "cut facts differ from charged cut prefix")
    _require(result.get("cut", {}).get("acquire") == [r["id"] for r in cut],
             "evidence.cut_order", "cut acquire list differs from charged prefix")
    scores = np.asarray(reference.helper_scores)
    bins = []
    for side in (0, 1):
        ids = np.flatnonzero(case.side == side)
        order = ids[np.lexsort((ids, scores[ids]))]
        for b in range(4):
            bins.append((side, b, tuple(map(int, order[len(order)*b//4:len(order)*(b+1)//4]))))
    domains = [tuple(j for j in ids if j not in cut_facts) for _, _, ids in bins]
    events = sum(bool(d) for d in domains) if audited else 0
    delta = Fraction(result.get("delta", "1/20"))
    _require(0 < delta <= Fraction(1, 20), "evidence.delta", "invalid cut failure budget")
    nominal = delta/events if events else None
    alpha = float(nominal) if nominal is not None else None
    if alpha is not None and Fraction.from_float(alpha) > nominal:
        alpha = math.nextafter(alpha, 0.)
    records = result.get("regions")
    _require(isinstance(records, list) and len(records) == 8, "evidence.regions_structure", "eight full-side cut bins required")
    sample = {r["id"]: r["label"] for r in audit}
    executed = Fraction(0)
    for (side, b, ids), domain, record in zip(bins, domains, records):
        drawn = [j for j in domain if j in sample]
        k = sum(sample[j] for j in drawn)
        known = sum(cut_facts.get(j, 0) for j in ids)
        lo, hi = count_interval_exact(len(domain), len(drawn), k, alpha) if domain and audited else (0, len(domain))
        a = str(Fraction.from_float(alpha)) if domain and audited else None
        if a is not None:executed += Fraction(a)
        expected = dict(side=side, bin=b, ids=list(ids), post_cut_unknown_ids=list(domain),
                        population=len(domain), sample=len(drawn), sample_ids=drawn, positive_sample=k,
                        cut_known_positive=known, unknown_lower=lo, unknown_upper=hi,
                        lower=known+lo, upper=known+hi, alpha_nominal=str(nominal) if domain and audited else None,
                        alpha_executed=a, alpha_float_hex=alpha.hex() if domain and audited else None)
        _require(record == expected, "evidence.regions_values", "cut bin does not match pinned helper order, paid cut, and fresh SRS")
    expected_confidence = dict(source="fresh_post_cut_srs" if audited else "point_facts_only",
                               budget_per_remaining_side=budget if audited else 0, event_count=events,
                               delta_nominal=str(delta), delta_executed=str(executed), side_draws=side_draws,
                               cut_facts_are_random_samples=False)
    _require(result.get("confidence") == expected_confidence, "evidence.cut_confidence", "post-cut SRS confidence ledger mismatch")


def strict_check_record(record, case, *, record_path, expected_key, expected_method,
                        expected_seed, expected_budget, expected_source_id,
                        expected_supplement_source_id=None, result_method_aliases=(),
                        seed0_control_seed=None, matched_result=None, reference=None):
    """Validate one completed record against explicit expected-cell arguments.

    matched_result, when supplied, is the matched original result (or its
    enclosing record). Its ordered audit ledger and complete ordered region
    records must match exactly. A reused PinnedReference must belong to the
    same case/version/identity mapping. Returns checks, never labels/tokens.
    """
    expected_seed = _integer(expected_seed, "argument.seed")
    expected_budget = _integer(expected_budget, "argument.budget")
    _require(Path(record_path).name == expected_key + ".json.gz", "cell.filename", "filename does not bind the expected key")
    _require(isinstance(record, Mapping), "cell.record", "record mapping required")
    for field, expected in (("key", expected_key), ("case_id", case.case_id), ("method", expected_method),
                            ("seed", expected_seed), ("budget", expected_budget), ("source_id", expected_source_id)):
        if field in ("seed", "budget"):
            _integer(record.get(field), "cell." + field)
        _require(record.get(field) == expected, "cell." + field, "expected-cell mismatch")
    if expected_supplement_source_id is None:
        _require("supplement_source_id" not in record, "cell.supplement_source_id", "unexpected supplement ID")
    else:
        _require(record.get("supplement_source_id") == expected_supplement_source_id,
                 "cell.supplement_source_id", "supplement source mismatch")
    _require(record.get("status") == "pass", "cell.status", "only completed passing records can be verified")
    result = record.get("result")
    _require(isinstance(result, Mapping), "cell.result", "result mapping required")
    methods = {expected_method, ORIGINAL_METHOD_ALIASES.get(expected_method, expected_method), *result_method_aliases}
    _require(result.get("method") in methods, "cell.result_method", "unlisted result method alias")
    for field in ("case_id", "dataset", "variant"):
        expected = case.case_id if field == "case_id" else getattr(case, field)
        if field in record:
            _require(record[field] == expected, "cell." + field, "case metadata mismatch")
        if field in result:
            _require(result[field] == expected, "cell.result_" + field, "result case metadata mismatch")
    actual_seed = expected_seed
    if seed0_control_seed is not None:
        _require(expected_seed == expected_budget == 0, "argument.control_seed", "seed convention applies only to declared 0/0 controls")
        actual_seed = _integer(seed0_control_seed, "argument.control_seed")
    if "seed" in result:
        _require(_integer(result["seed"], "cell.result_seed") == actual_seed, "cell.result_seed", "result seed mismatch")
    else:
        _require(expected_seed == expected_budget == 0 and result.get("no_audit") is True,
                 "cell.result_seed", "only exact 0/0 no-audit controls may omit their seed")
    if "audit_budget" in result:
        _require(_integer(result["audit_budget"], "cell.result_budget") == expected_budget, "cell.result_budget", "result audit budget mismatch")
    elif expected_budget:
        _require(False, "cell.result_budget", "audited cell must record audit budget")
    reference = reference or load_pinned_reference(case)
    _require(isinstance(reference, PinnedReference) and reference.case_id == case.case_id
             and reference.target_version == case.target_version and reference.oracle_version == case.oracle_version
             and reference.identities == tuple(case.ids) and reference.row_predicates == _case_pairs(case),
             "reference.case", "reference belongs to another case/version/identity mapping")
    _require(all(case.metadata.get("source_hashes", {}).get(name) == digest for name, digest in reference.source_hashes),
             "reference.case", "case pins differ from the joined reference")
    ledger = result.get("oracle_ledger", result.get("ledger"))
    helper = result.get("helper_ledger")
    _validate_ledger(ledger, case, reference)
    _validate_ledger(helper, case, reference, helper=True)
    facts = result.get("facts")
    _require(isinstance(facts, Mapping), "facts.ledger", "fact mapping required")
    normalized = {}
    for raw, value in facts.items():
        if isinstance(raw, str):
            _require(raw.isdecimal() and str(int(raw)) == raw, "facts.id", "canonical JSON integer key required")
            raw = int(raw)
        j = _integer(raw, "facts.id")
        _require(j < case.n and j not in normalized, "facts.id", "invalid or duplicate fact identity")
        normalized[j] = _integer(value, "facts.label")
    _require(normalized == {r["id"]: r["label"] for r in ledger}, "facts.ledger", "facts differ from charged labels")
    summary = result.get("summary", result)
    _validate_summary(summary, ledger, helper)
    # Original exact-control results also expose flat summaries. Validate
    # those independently when present, rather than accepting divergent copies.
    if summary is not result and "oracle_requests" in result:
        _validate_summary(result, ledger, helper)
    audit = [r for r in ledger if r["phase"] == "audit"]
    cut = [r for r in ledger if r["phase"] == "cut"]
    is_cut = str(result.get("method", "")).startswith("cut_")
    if is_cut:
        _require([r["phase"] for r in ledger] == ["cut"]*len(cut)+["audit"]*len(audit)+["execution"]*(len(ledger)-len(cut)-len(audit)),
                 "evidence.cut_phase_order", "cut, fresh audit, execution order required")
        _require({r["id"] for r in helper} == set(range(case.n)), "evidence.cut_helper", "complete helper scores must be charged")
    else:
        _require(ledger[:len(audit)] == audit, "evidence.audit_prefix", "audit must precede execution/cut")
    rng = np.random.default_rng(actual_seed)
    drawn, side_draws = [], []
    cut_ids = {r["id"] for r in cut} if is_cut else set()
    for side in (0, 1):
        population = [j for j in range(case.n) if int(case.side[j]) == side and j not in cut_ids]
        selected = list(map(int, rng.choice(population, size=min(expected_budget, len(population)), replace=False)))
        drawn.extend(selected)
        side_draws.append(dict(side=side, population=len(population), sample=len(selected), sample_ids=selected))
    _require([r["id"] for r in audit] == drawn, "evidence.audit_order", "audit ledger differs from declared ordered side SRS")
    if matched_result is not None:
        if "result" in matched_result:
            matched_result = matched_result["result"]
        prior = matched_result.get("oracle_ledger", matched_result.get("ledger", []))
        _require(audit == [r for r in prior if r["phase"] == "audit"], "evidence.audit_match", "ordered audit differs from matched original")
        _require(result.get("regions", []) == matched_result.get("regions", []),
                 "evidence.regions_match", "ordered region evidence differs from matched original")
    if is_cut:
        _validate_cut_regions(case, result, audit, cut, reference,
                              audited=str(result["method"]).endswith("_audit"), budget=expected_budget, side_draws=side_draws)
    else:
        _validate_regions(case, result, audit, required=result.get("no_audit") is not True)
    # Keep the historical SQL/count/coverage/affine checker as a separate layer.
    base = dict(record); base_result = dict(result); base_result["summary"] = summary; base["result"] = base_result
    try:
        check = check_record(base, case, reference.labels)
    except (AssertionError, KeyError, ValueError, TypeError) as exc:
        raise StrictValidationError("base.validation", str(exc)) from exc
    _require(check.get("status") == "pass", "base.validation", "base validator did not pass")
    return {**check, "strict_validation": True, "pinned_oracle_charge_checks": len(ledger),
            "pinned_helper_charge_checks": len(helper), "pinned_joined_identities": case.n,
            "record_cell_bound": True, "ordered_audit_verified": True,
            "matched_original_evidence_verified": matched_result is not None,
            "pinned_reference_hashes": dict(reference.source_hashes)}
