"""Independent small-instance references for paid-cut external study arms.

No external study cell is run. Expected supports enumerate Boolean worlds;
expected bags enumerate (key,left,right) derivations with shared-key repeats.
"""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_tiny_case, make_join
from cut_arms import (AffineEnvelope, InconsistentEvidence, DELTA,
                      downward_alpha, run_cut, score_bins)


def bag(case, labels, discarded=()):
    drop = set(discarded)
    return {(k, a, b) for k, (left, right) in enumerate(zip(case.key_left, case.key_right))
            for a in left for b in right
            if labels[a] and labels[b] and a not in drop and b not in drop}


def validate_result(case, truth, result):
    facts = result["facts"]
    full, expected = bag(case, truth), bag(case, [facts.get(j, 0) for j in range(case.n)])
    assert expected == bag(case, truth, result["discarded"])
    assert result["output"]["returned_count"] == len(expected)
    assert result["output"]["full_count"] == len(full)
    assert result["output"]["bag_subset"]
    assert result["affine_returned_count"] == len(expected)
    paid = result["oracle_ledger"]
    assert len({r["id"] for r in paid}) == len(paid)
    assert {r["id"]: r["label"] for r in paid} == facts
    assert result["summary"]["oracle_requests"] == len(paid)
    assert result["summary"]["helper_requests"] == case.n
    assert result["summary"]["initial_cache_count"] == 0
    for phase in ("cut", "audit", "execution"):
        assert result["summary"]["phase_counts"][phase] == sum(r["phase"] == phase for r in paid)
    cut_ids = set(result["cut"]["acquire"])
    audit_ids = {r["id"] for r in paid if r["phase"] == "audit"}
    assert not cut_ids & audit_ids
    assert cut_ids == {r["id"] for r in paid if r["phase"] == "cut"}
    assert Fraction(result["cut"]["cost"]) == len(cut_ids)
    covered = True
    for region in result["regions"]:
        ids, domain = region["ids"], region["post_cut_unknown_ids"]
        assert set(domain) == set(ids) - cut_ids
        assert region["cut_known_positive"] == sum(truth[j] for j in ids if j in cut_ids)
        assert region["lower"] == region["cut_known_positive"] + region["unknown_lower"]
        assert region["upper"] == region["cut_known_positive"] + region["unknown_upper"]
        assert region["sample"] == len(set(domain) & audit_ids)
        assert region["positive_sample"] == sum(truth[j] for j in set(domain) & audit_ids)
        covered &= region["lower"] <= sum(truth[j] for j in ids) <= region["upper"]
        if region["alpha_executed"]:
            assert Fraction(region["alpha_executed"]) <= Fraction(region["alpha_nominal"])
    assert Fraction(result["confidence"]["delta_executed"]) <= DELTA
    for draw in result["confidence"]["side_draws"]:
        population = {j for j in range(case.n) if case.side[j] == draw["side"] and j not in cut_ids}
        assert draw["population"] == len(population)
        assert draw["sample"] == min(result["audit_budget"], len(population))
        assert len(set(draw["sample_ids"])) == draw["sample"]
        assert set(draw["sample_ids"]) <= population
    if covered:
        assert 10 * (len(full) - len(expected)) <= len(full)
        assert result["final_scaled_support"] is not None
        assert result["final_scaled_support"] <= 0
    return covered


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(261016)
    support_checks = completion_checks = executions = coverage_failures = 0
    negative_maxima = 0
    # Boolean-sharp support, independent of the cut solver and HG implementation.
    for repeat in range(60):
        n = 10
        truth = rng.integers(0, 2, n).tolist()
        scores = rng.choice([0., .5, 1.], n)
        case = make_tiny_case(side=[0]*5+[1]*5, key_left=((0, 1, 2), (2, 3, 4)),
                              key_right=((5, 6, 7), (7, 8, 9)), truth=truth,
                              scores=scores, name=f"support{repeat}")
        bins = score_bins(case, scores)
        cut = {j for j in range(n) if rng.random() < .2}
        weights = {j: int(rng.integers(0, 10)) for j in range(n) if j not in cut}
        facts = {j: truth[j] for j in range(n) if j in cut or rng.random() < .25}
        constant = int(rng.integers(0, 20))
        regions = []
        for side, b, ids in bins:
            positive = sum(truth[j] for j in ids)
            regions.append({"side": side, "bin": b, "ids": ids,
                            "lower": int(rng.integers(0, positive+1)),
                            "upper": int(rng.integers(positive, len(ids)+1))})
        order = rng.permutation(n)
        env = AffineEnvelope(n, constant, weights, regions, order)
        worlds = [z for z in product((0, 1), repeat=n)
                  if all(z[j] == y for j, y in facts.items()) and
                  all(r["lower"] <= sum(z[j] for j in r["ids"]) <= r["upper"] for r in regions)]
        assert worlds
        for cardinality in range(n+1):
            drop = set(map(int, order[:cardinality])) - facts.keys()
            reference = max(10*sum(w*z[j] for j, w in weights.items() if j in drop) -
                            constant - sum(w*z[j] for j, w in weights.items()) for z in worlds)
            actual = env.support(facts, cardinality)
            assert actual == reference
            support_checks += 1
            negative_maxima += actual < 0
        case.close_sql()
    # Explicit infeasible intervals must raise, rather than certify an empty set.
    bad = AffineEnvelope(1, 0, {0: 1}, [{"side": 0, "bin": 0, "ids": [0],
                                         "lower": 1, "upper": 1}], [0])
    try:
        bad.support({0: 0}, 1)
        raise AssertionError("Empty evidence was accepted")
    except InconsistentEvidence:
        pass
    # Every truth and every post-cut Boolean completion of a repeated-key graph.
    for truth in product((0, 1), repeat=6):
        case = make_tiny_case(side=[0]*3+[1]*3,
            key_left=((0, 1), (1, 2), (0, 2)),
            key_right=((3, 4), (4, 5), (3, 5)), truth=truth,
            scores=[0., .7, 1., .2, .9, .3], name="multikey_execution")
        join = make_join(case)
        plan = join.plan_cut({})
        cut_facts = {j: truth[j] for j in plan.acquire}
        affine = join.linearize(cut_facts)
        unknown = sorted(affine.weight_map())
        for completion in product((0, 1), repeat=len(unknown)):
            labels = dict(cut_facts); labels.update(zip(unknown, completion))
            expected = len(bag(case, labels))
            assert expected == affine.constant + sum(w*labels[j] for j, w in affine.weights)
            completion_checks += 1
        for audit in (False, True):
            arms = [run_cut(case, 101, 1, audit=audit, feedback=fb) for fb in (False, True)]
            for result in arms:
                coverage_failures += not validate_result(case, truth, result)
                executions += 1
            assert arms[0]["cut_facts"] == arms[1]["cut_facts"]
            assert arms[0]["confidence"] == arms[1]["confidence"]
            assert arms[0]["initial_mask_index"] == arms[1]["initial_mask_index"]
            static_paid = {r["id"] for r in arms[0]["oracle_ledger"]}
            feedback_paid = {r["id"] for r in arms[1]["oracle_ledger"]}
            assert feedback_paid <= static_paid
        case.close_sql()
    # Nonmatching keys, isolated nodes, helper ties, and zero answers.
    fixtures = [
        ([1, 0, 1, 0, 0, 1, 1, 0], ((0, 1), (2,), (3,)), ((4, 5), (6,), ())),
        ([0]*8, ((0, 1), (1, 2)), ((4, 5), (5, 6))),
        ([1]*8, ((0,), (1,)), ((), ())),
    ]
    for number, (truth, left, right) in enumerate(fixtures):
        case = make_tiny_case(side=[0]*4+[1]*4, key_left=left, key_right=right,
                              truth=truth, scores=[.5]*8, name=f"edge{number}")
        for budget in (0, 1, 2, 125, 500):
            result = run_cut(case, 202, budget)
            coverage_failures += not validate_result(case, truth, result)
            executions += 1
            # Every tied score bin is ordered by integer identity.
            for side in (0, 1):
                flattened = [j for r in result["regions"] if r["side"] == side for j in r["ids"]]
                assert flattened == [j for j in range(case.n) if case.side[j] == side]
        case.close_sql()
    # A disconnected pair of unequal bicliques forces a mixed-side cut.
    # Three-member full bins then contain both cut facts and a genuinely
    # partial fresh draw, exercising conditional within-bin HG semantics.
    mixed_cut_audit_checks = 0
    for repeat in range(4):
        truth = rng.integers(0, 2, 24).tolist()
        case = make_tiny_case(side=[0]*12+[1]*12,
            key_left=(tuple(range(8)), tuple(range(8, 12))),
            key_right=(tuple(range(12, 15)), tuple(range(15, 24))), truth=truth,
            scores=rng.permutation(24)/24, name=f"mixed_cut{repeat}")
        for budget in (1, 4, 125, 500):
            result = run_cut(case, 303, budget)
            coverage_failures += not validate_result(case, truth, result)
            cut_sides = {int(case.side[j]) for j in result["cut"]["acquire"]}
            assert cut_sides == {0, 1}
            mixed_cut_audit_checks += sum(0 < r["sample"] < r["population"]
                                          for r in result["regions"])
            executions += 1
        case.close_sql()
    assert mixed_cut_audit_checks > 0
    # One small >128 residual fixture actually exercises mask feedback after a
    # full batch. This is synthetic, not a study data cell or a performance claim.
    truth = [1]*320
    case = make_tiny_case(side=[0]*20+[1]*300, key_left=(tuple(range(20)),),
                          key_right=(tuple(range(20, 320)),), truth=truth,
                          scores=np.linspace(0, 1, 320), name="batch_feedback")
    paired = [run_cut(case, 303, 125, audit=False, feedback=fb) for fb in (False, True)]
    for result in paired:
        validate_result(case, truth, result); executions += 1
    assert paired[1]["final_mask_index"] > paired[1]["initial_mask_index"]
    assert {r["id"] for r in paired[1]["oracle_ledger"]} <= {r["id"] for r in paired[0]["oracle_ledger"]}
    batch_diagnostic = [{"method": r["method"], "summary": r["summary"],
                         "initial_mask_index": r["initial_mask_index"],
                         "final_mask_index": r["final_mask_index"], "output": r["output"]}
                        for r in paired]
    case.close_sql()
    for events in range(1, 1000):
        assert Fraction.from_float(downward_alpha(DELTA/events)) <= DELTA/events
    source = Path(__file__).with_name("cut_arms.py")
    checks = Path(__file__)
    out = {"status": "pass", "exact_support_boolean_checks": support_checks,
           "negative_signed_maximum_checks": negative_maxima,
           "post_cut_all_boolean_completion_checks": completion_checks,
           "complete_synthetic_sql_executions": executions,
           "mixed_side_cut_partial_region_audit_checks": mixed_cut_audit_checks,
           "observed_joint_region_coverage_failures": coverage_failures,
           "downward_budget_checks": 999, "explicit_empty_evidence_rejection": True,
           "batch_feedback_diagnostic": batch_diagnostic,
           "scope": "independent small synthetic checks; no external matrix cell executed",
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "checks_sha256": hashlib.sha256(checks.read_bytes()).hexdigest(),
           "elapsed_seconds": time.perf_counter()-started}
    Path(__file__).with_name("cut_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
