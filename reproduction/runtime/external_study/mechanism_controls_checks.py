"""Synthetic checks of post-matrix exploratory mechanism controls only.

These checks execute no external matrix. For every feasible completion,
q*loss-p*total <= q*Lmax-p*Tmin, since q,p are nonnegative. Summing per-key
extrema preserves this inequality under the original overlap relaxation.
The finite Boolean checks below verify the implementation against that claim.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Oracle, make_tiny_case, possible_unknowns, potential_degrees
from direct_arms import ALLOWANCE, Envelope, InconsistentEvidence, RegionSide, key_support, make_regions, run_direct
import mechanism_controls as controls


def _points(side):
    return [(a, b) for a in range(side.unknown_lost + 1)
            for b in range(side.known_positive, side.known_positive + side.unknown_kept + 1)
            if side.lower_total <= a + b <= side.upper_total]


def check_support_arithmetic():
    sides = []
    for positive, lost, kept in product(range(2), repeat=3):
        capacity = positive + lost + kept
        for lower in range(capacity + 1):
            for upper in range(lower, capacity + 1):
                side = RegionSide(lower, upper, positive, lost, kept)
                if _points(side):
                    sides.append(side)
    pairs = strict = 0
    p, q = ALLOWANCE.numerator, ALLOWANCE.denominator
    for left, right in product(sides, repeat=2):
        objectives, losses, totals = [], [], []
        for (a, b), (c, d) in product(_points(left), _points(right)):
            total, retained = (a + b) * (c + d), b * d
            objectives.append(q * (total - retained) - p * total)
            losses.append(total - retained)
            totals.append(total)
        original = key_support(left, right, ALLOWANCE)
        separate = controls.separate_extrema_components(((left, right),))
        assert original == max(objectives)
        assert separate["lost_max"] == max(losses)
        assert separate["total_min"] == min(totals)
        assert separate["scaled_support"] == q * max(losses) - p * min(totals)
        assert separate["scaled_support"] >= original
        strict += separate["scaled_support"] > original
        pairs += 1
    assert strict > 0
    # A wide interval and no unknown discards must still certify the empty mask.
    wide = RegionSide(0, 20, 0, 0, 20)
    assert controls.separate_extrema_components(((wide, wide),))["scaled_support"] == 0
    assert controls.separate_extrema_components(())["scaled_support"] == 0
    # Empty domains must raise even if the other side forces an empty answer.
    invalid = (RegionSide(2, 1, 0, 1, 1), RegionSide(0, 0, 1, 0, 0),
               RegionSide(3, 3, 0, 1, 1))
    empty = RegionSide(0, 0, 0, 0, 0)
    for bad in invalid:
        for pair in ((bad, empty), (empty, bad)):
            try:
                controls.separate_extrema_components((pair,))
            except InconsistentEvidence:
                pass
            else:
                raise AssertionError("An empty domain was silently certified")
    return {"side_domains": len(sides), "exact_pair_enumerations": pairs,
            "strictly_looser_separate_bounds": strict, "inconsistent_pairs_rejected": 6}


def _order(case):
    rank = potential_degrees(case, {})
    return np.lexsort((np.arange(case.n), rank)), np.lexsort((np.arange(case.n), -rank))


def _bag(case, labels):
    return sorted((key, left, right)
                  for key, pair in enumerate(zip(case.key_left, case.key_right))
                  for left in pair[0] for right in pair[1]
                  if labels.get(left) == 1 and labels.get(right) == 1)


def check_boolean_envelopes():
    predictions = completions = 0
    for overlap in (False, True):
        left = ((0,), (1,)) if not overlap else ((0, 1), (0,))
        right = ((2,), (3,)) if not overlap else ((2,), (2, 3))
        case = make_tiny_case(side=(0, 0, 1, 1), key_left=left, key_right=right,
                              truth=(0, 0, 0, 0), name=f"post_matrix_support_{overlap}")
        order, _ = _order(case)
        groups = tuple(dict.fromkeys(tuple(ids) for pair in zip(left, right) for ids in pair))
        for truth in product((0, 1), repeat=4):
            for cached in ((), (0, 2), (0, 1, 2, 3)):
                facts = {j: truth[j] for j in cached}
                for tight in (False, True):
                    intervals = {ids: ((sum(truth[j] for j in ids),) * 2 if tight else (0, len(ids))) for ids in groups}
                    original = Envelope(case, intervals, order)
                    separate = controls.SeparateExtremaEnvelope(case, intervals, order)
                    for cardinality in range(5):
                        signed = original.support(facts, cardinality)
                        bound = separate.support(facts, cardinality)
                        assert bound >= signed
                        if cardinality == 0:
                            assert bound <= 0
                        base_discard = set(int(j) for j in order[:cardinality])
                        compatible = []
                        for candidate in product((0, 1), repeat=4):
                            if any(candidate[j] != value for j, value in facts.items()):
                                continue
                            if any(not lo <= sum(candidate[j] for j in ids) <= hi
                                   for ids, (lo, hi) in intervals.items()):
                                continue
                            labels = dict(enumerate(candidate))
                            full = len(_bag(case, labels))
                            retained_labels = {j: value for j, value in labels.items()
                                               if j not in base_discard or facts.get(j) == 1}
                            lost = full - len(_bag(case, retained_labels))
                            compatible.append(10 * lost - full)
                            completions += 1
                        assert compatible and signed >= max(compatible)
                        assert bound >= max(compatible)
                        predictions += 1
        case.close_sql()
    return {"mask_fact_interval_predictions": predictions, "feasible_boolean_completions": completions}


def naive_fixed_pruning(case, seed, budget):
    oracle = Oracle(case)
    intervals, regions, _ = make_regions(case, oracle, seed, budget)
    order, execution = _order(case)
    cards = [case.n * step // 64 for step in range(65)]
    index = Envelope(case, intervals, order).advance(oracle.facts, 0, cards)
    retained = np.ones(case.n, dtype=bool)
    retained[order[:cards[index]]] = False
    while True:
        possible = set(possible_unknowns(case, oracle.facts, retained))
        selected = [int(j) for j in execution if int(j) in possible][:128]
        if not selected:
            break
        oracle.acquire(selected, "execution")
    return oracle, index, regions


def _check_output(case, result, truth):
    expected = _bag(case, dict(enumerate(truth)))
    returned = _bag(case, result["facts"])
    out = result["output"]
    assert out["full_count"] == len(expected) and out["returned_count"] == len(returned)
    assert out["bag_subset"] and out["lost_count"] == len(expected) - len(returned)
    actual = case._sql["connection"].execute(
        "SELECT key_id,left_id,right_id FROM returned_derivations ORDER BY ALL"
    ).fetchall()
    assert actual == returned
    covered = all(region["lower"] <= sum(truth[j] for j in region["ids"]) <= region["upper"] for region in result["regions"])
    if covered and not result.get("conflict_freeze", False):
        assert 10 * out["lost_count"] <= out["full_count"]


def check_matched_executors():
    cases = policy_runs = 0
    for fixture in range(2):
        left, right = (((0,), (1,)), ((2,), (3,))) if fixture == 0 else (((0, 1), (0,)), ((2,), (2, 3)))
        for truth in product((0, 1), repeat=4):
            for seed in (101, 202, 303):
                case = make_tiny_case(side=(0, 0, 1, 1), key_left=left, key_right=right,
                                      truth=truth, scores=(.1, .9, .3, .7),
                                      name=f"post_matrix_executor_{fixture}_{truth}_{seed}")
                try:
                    baseline = run_direct(case, seed, 1, helper=False)
                    _check_output(case, baseline, truth)
                    keywise = run_direct(case, seed, 1, helper=False, keywise=True)
                    _check_output(case, keywise, truth)
                    audit_ledger = [row for row in baseline["oracle_ledger"] if row["phase"] == "audit"]
                    audit_facts = {row["id"]: row["label"] for row in audit_ledger}
                    naive, frozen_index, frozen_regions = naive_fixed_pruning(case, seed, 1)
                    for method in controls.METHODS:
                        result = controls.run_mechanism(case, method, seed, 1)
                        policy_runs += 1
                        _check_output(case, result, truth)
                        summary = result["summary"]
                        assert summary["helper_requests"] == summary["helper_read_calls"] == summary["helper_read_items"] == 0
                        assert result["helper_ledger"] == []
                        assert result["regions"] == baseline["regions"] == frozen_regions
                        assert result["initial_facts_after_audit"] == audit_facts
                        assert [row for row in result["oracle_ledger"] if row["phase"] == "audit"] == audit_ledger
                        assert len(result["facts"]) == len(result["oracle_ledger"]) == summary["oracle_requests"]
                        assert len({row["id"] for row in result["oracle_ledger"]}) == summary["oracle_requests"]
                        assert result["initial_R0"] == result["initial_retained_unknown"]
                        assert result["post_matrix_control"] and result["exploratory"] and not result["preregistered"]
                        assert result["initial_certificate"]["fact_count"] == len(audit_facts)
                        assert result["final_certificate"]["fact_count"] == len(result["facts"])
                        assert result["final_certificate"]["possible_retained_unknown_count"] == 0
                        if method == "keywise_degree":
                            for field in ("oracle_ledger", "regions", "facts", "initial_mask_index", "final_mask_index", "initial_retained_unknown", "discarded"):
                                assert result[field] == keywise[field], field
                            assert [(row["paid"], row["mask_index"], row["discard_cardinality"]) for row in result["events"]] == [
                                (row["paid"], row["mask_index"], row["discard_cardinality"]) for row in keywise["events"]]
                        elif method == "fixed_mask_pruning_degree":
                            assert result["oracle_ledger"] == naive.ledger
                            assert result["initial_mask_index"] == result["final_mask_index"] == frozen_index == baseline["initial_mask_index"]
                            assert result["initial_retained_unknown"] == baseline["initial_retained_unknown"]
                            assert all(row["mask_index"] == frozen_index for row in result["events"])
                            order, _ = _order(case)
                            assert set(result["discarded"]) == set(map(int, order[:result["initial_discard_cardinality"]])) - audit_facts.keys()
                        else:
                            assert result["initial_mask_index"] <= baseline["initial_mask_index"]
                    cases += 1
                finally:
                    case.close_sql()
    return {"matched_boolean_case_seed_pairs": cases, "new_materialized_policy_runs": policy_runs,
            "original_reference_materializations": 2 * cases}


def check_pruning_and_conflicts():
    side = [0] * 130 + [1] * 130
    left, right = (tuple(range(130)),), (tuple(range(130, 260)),)
    zero_left_truth = [0] * 130 + [1] * 130
    case = make_tiny_case(side=side, key_left=left, key_right=right, truth=zero_left_truth,
                          name="post_matrix_frozen_mask_pruning")
    try:
        fixed = controls.run_mechanism(case, "fixed_mask_pruning_degree", 101, 0)
        _check_output(case, fixed, zero_left_truth)
        original_static = run_direct(case, 101, 0, helper=False, feedback=False)
        assert fixed["initial_mask_index"] == fixed["final_mask_index"] == 0
        assert fixed["summary"]["oracle_requests"] == 256 < original_static["summary"]["oracle_requests"] == 260
        assert all(event["mask_index"] == 0 for event in fixed["events"])
        pruning_example = {"identities": 260, "fixed_mask_with_pruning_requests": 256,
                           "original_static_without_pruning_requests": 260,
                           "initial_mask_index": 0, "final_mask_index": 0}
    finally:
        case.close_sql()
    initial_conflicts = later_conflicts = 0
    for when in ("initial", "feedback"):
        truth = [1] * 260
        case = make_tiny_case(side=side, key_left=left, key_right=right, truth=truth,
                              name=f"post_matrix_conflict_{when}")
        def conflicting_regions(target, oracle, seed, budget):
            intervals, regions, seconds = make_regions(target, oracle, seed, budget)
            intervals[target.key_left[0]] = (131, 131) if when == "initial" else (0, 1)
            return intervals, regions, seconds
        try:
            with patch.object(controls, "make_regions", conflicting_regions):
                for method in controls.METHODS:
                    result = controls.run_mechanism(case, method, 101, 0)
                    _check_output(case, result, truth)
                    assert result["conflict_freeze"] and result["conflicts"]
                    assert result["initial_mask_index"] == result["final_mask_index"] == 0
                    assert result["final_scaled_support"] is None and result["final_certified_cardinality"] is None
                    assert result["output"]["exact"]
                    if when == "initial":
                        assert result["initial_certified_cardinality"] is None
                        assert result["conflicts"][0]["stage"] == "initial_projection"
                        initial_conflicts += 1
                    else:
                        assert result["conflicts"][0]["stage"] == "execution_feedback"
                        assert result["conflicts"][0]["paid"] == 128
                        later_conflicts += 1
        finally:
            case.close_sql()
    return {"pruning_example": pruning_example, "initial_conflicts_frozen": initial_conflicts,
            "feedback_conflicts_frozen": later_conflicts, "extra_materialized_policy_runs": 7}


def main():
    started = time.perf_counter()
    report = {"status": "passed", "post_matrix_control": True, "exploratory": True,
              "preregistered": False, "registration_note": controls.REGISTRATION_NOTE,
              "scope": "Synthetic correctness checks only; no external matrix executed",
              "support_arithmetic": check_support_arithmetic(),
              "boolean_envelopes": check_boolean_envelopes(),
              "matched_executors": check_matched_executors(),
              "pruning_and_conflicts": check_pruning_and_conflicts(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ("mechanism_controls.py", "mechanism_controls_checks.py", "direct_arms.py", "common.py")},
              "elapsed_seconds": time.perf_counter() - started}
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
