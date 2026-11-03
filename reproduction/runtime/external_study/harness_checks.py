"""Small independent correctness checks; this script is not a study matrix."""
from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
import json
import sys
import time

import numpy as np

try:
    from .common import (Case, Oracle, evaluate, load_cases, make_tiny_case,
        possible_unknowns, potential_degrees)
    from .exact_baselines import LivePartners, METHODS, run_exact
except ImportError:
    from common import (Case, Oracle, evaluate, load_cases, make_tiny_case,
        possible_unknowns, potential_degrees)
    from exact_baselines import LivePartners, METHODS, run_exact

HERE = Path(__file__).resolve().parent


def brute_bag(case, truth):
    # Independent explicit pair enumeration; every shared key is retained.
    return Counter((k, i, j) for k in range(len(case.key_names))
        for i in case.key_left[k] for j in case.key_right[k]
        if truth[i] == truth[j] == 1)


def fixture(truth, name):
    return make_tiny_case(side=(0, 0, 0, 1, 1, 1),
        key_names=("a", "b", "left-only", "right-only"),
        key_left=((0, 1), (0,), (2,), ()), key_right=((3, 4), (3,), (), (5,)),
        truth=truth, scores=(0., .25, 1., .5, .75, 0.), name=name,
        tokens={"oracle_input_tokens": np.arange(10, 16), "oracle_output_tokens": np.arange(20, 26),
                "helper_input_tokens": np.arange(30, 36), "helper_output_tokens": np.arange(40, 46)})


def slow_greedy(case, initial=None):
    oracle = Oracle(case, initial)
    scores = oracle.use_scores(range(case.n))
    while True:
        possible = possible_unknowns(case, oracle.facts)
        if not possible: break
        degree = potential_degrees(case, oracle.facts, opposite_scores=scores)
        j = min(possible, key=lambda v: (-scores[v] * degree[v], v))
        oracle.acquire((j,), "execution")
    return [r["id"] for r in oracle.ledger]


def tiny_checks():
    counts = {"exact_policy_replays": 0, "materialized_bag_comparisons": 0,
              "incremental_degree_states": 0, "lazy_greedy_order_comparisons": 0,
              "affine_decomposition_checks": 0}
    adverse_or_empty = 0
    for index, active in enumerate(product((0, 1), repeat=4)):
        truth = (active[0], active[1], 1, active[2], active[3], 1)
        case = fixture(truth, f"exhaustive-{index}")
        expected = brute_bag(case, truth)
        for method in METHODS:
            result = run_exact(case, method)
            assert result["output"]["exact"] and result["output"]["full_count"] == sum(expected.values())
            actual = Counter(case._sql["connection"].execute("SELECT key_id,left_id,right_id FROM returned_derivations").fetchall())
            assert actual == expected
            assert result["phase_counts"]["audit"] == 0
            assert len({r["id"] for r in result["ledger"]}) == result["oracle_requests"]
            assert not ({2, 5} & {r["id"] for r in result["ledger"]})
            if method != "helper_greedy": assert result["helper_requests"] == 0
            if method == "exact_scan": assert result["oracle_requests"] == 4
            if method == "cover":
                d = result["diagnostics"]
                affine = d["residual_constant"] + sum(w * truth[j] for j, w in d["residual_weights"])
                assert affine == sum(expected.values())
                assert d["cut_verification"]["zero_primal_dual_gap"]
                counts["affine_decomposition_checks"] += 1
            if method == "helper_greedy":
                assert [r["id"] for r in result["ledger"]] == slow_greedy(case)
                counts["lazy_greedy_order_comparisons"] += 1
            counts["exact_policy_replays"] += 1; counts["materialized_bag_comparisons"] += 1
        if not expected: adverse_or_empty += 1
        oracle = Oracle(case)
        scores = oracle.use_scores(range(case.n))
        live = LivePartners(case, oracle.facts, scores)
        for j in (0, 3, 1, 4, 2, 5):
            np.testing.assert_array_equal(live.degree, potential_degrees(case, oracle.facts).astype(np.int64))
            np.testing.assert_array_equal(live.weighted_degree, potential_degrees(case, oracle.facts, opposite_scores=scores))
            counts["incremental_degree_states"] += 1
            live.observe(oracle.acquire((j,), "execution"))
        case.close_sql()
    # Duplicate across two keys must survive (0,3 appears twice in the bag).
    case = fixture((1, 1, 1, 1, 1, 1), "partial-bag")
    partial = evaluate(case, {0: 1, 3: 1})
    assert (partial["full_count"], partial["returned_count"], partial["lost_count"]) == (5, 2, 3)
    assert partial["bag_subset"] and partial["loss"] == 3 / 5
    # Initial facts are free cache, not newly paid audit observations.
    for initial in ({0: 1}, {0: 1, 3: 1}):
        for method in METHODS:
            result = run_exact(case, method, initial_facts=initial)
            assert result["output"]["exact"]
            assert all(r["id"] not in initial for r in result["ledger"])
            counts["exact_policy_replays"] += 1; counts["materialized_bag_comparisons"] += 1
    # Independent minimum vertex-cover search on expanded metadata edges.
    edges = {(i, j) for a, b in zip(case.key_left, case.key_right) for i in a for j in b}
    optimum = min(sum(mask) for mask in product((0, 1), repeat=case.n) if all(mask[i] or mask[j] for i, j in edges))
    cover = run_exact(case, "cover")
    assert cover["diagnostics"]["cut_cost"] == {"numerator": optimum, "denominator": 1}
    case.close_sql()
    # Ledger/token checks include duplicate requests, repeated helper reads,
    # separate phases, and truthful initial cache exclusion.
    o = Oracle(case, {0: 1})
    assert o.acquire((0, 1, 1), "audit") == {1: 1}
    assert o.acquire((1, 3), "cut") == {3: 1}
    assert o.acquire((4,), "execution") == {4: 1}
    assert o.use_scores((2, 2, 1)).tolist() == [1., 1., .25]
    o.use_scores((1, 5))
    summary = o.summary()
    assert summary["oracle_requests"] == 3 and summary["phase_counts"] == {"audit": 1, "cut": 1, "execution": 1}
    assert summary["oracle_tokens_in"] == 11 + 13 + 14
    assert summary["oracle_tokens_out"] == 21 + 23 + 24
    assert summary["helper_requests"] == 3 and summary["helper_read_items"] == 5
    assert summary["helper_tokens_in"] == 31 + 32 + 35
    assert summary["helper_tokens_out"] == 41 + 42 + 45
    for bad in ((-1,), (6,), (True,), (1.5,)):
        try: o.acquire(bad, "execution")
        except ValueError: pass
        else: raise AssertionError("Invalid identity accepted")
    try: Oracle(case, {0: 0})
    except ValueError: pass
    else: raise AssertionError("Contradictory cache accepted")
    try:
        make_tiny_case(side=(0, 1), key_left=((0,),), key_right=((1,),), truth=(1, 1), ids=("same", "same"))
    except ValueError: pass
    else: raise AssertionError("Cross-side oracle alias accepted")
    try:
        make_tiny_case(side=(0, 1), key_left=((0, 0),), key_right=((1,),), truth=(1, 1))
    except ValueError: pass
    else: raise AssertionError("Duplicate key incidence accepted")
    # No matching incidence on either side, including an all-positive tape.
    empty = make_tiny_case(side=(0, 1), key_left=((0,), ()), key_right=((), (1,)), truth=(1, 1))
    for method in METHODS:
        result = run_exact(empty, method)
        assert result["oracle_requests"] == 0 and result["output"]["full_count"] == 0 and result["output"]["loss"] == 0
        counts["exact_policy_replays"] += 1; counts["materialized_bag_comparisons"] += 1
    empty.close_sql()
    counts.update({"empty_truth_cases_retained": adverse_or_empty, "ledger_tokens": "passed",
                   "identity_and_incidence_rejections": "passed", "partial_materialization": "5 full / 2 returned / 3 lost", "minimum_cover_bruteforce": optimum})
    return counts


def production_metadata_and_smoke():
    started = time.perf_counter(); cases = load_cases()
    metadata_seconds = time.perf_counter() - started
    assert len(cases) == 32 and len({c.case_id for c in cases}) == 32
    for case in cases:
        assert case.n == 5000 and case.side.dtype == np.int8 and case.scores.dtype == np.float64
        assert tuple(case.row_ids) == tuple(sorted(set(case.row_ids)))
        assert list(case.side).count(0) == list(case.side).count(1) == 2500
        # Rebuild each key from public row metadata independently of loader.
        expected = {}
        for j, meta in enumerate(case.row_metadata):
            if case.dataset == "dbpedia":
                levels = ("l1",) if case.variant == "single" else ("l1", "l2", "l3")
                keys = tuple((level, meta[level]) for level in levels if meta[level] is not None)
            else:
                annotations = tuple(sorted(set(meta["emotions"])))
                keys = annotations[:1] or ("__EMPTY_ANNOTATION__",) if case.variant == "single" else annotations
            for key in keys: expected.setdefault(key, ([], []))[int(case.side[j])].append(j)
        assert case.key_names == tuple(sorted(expected))
        assert case.key_left == tuple(tuple(expected[k][0]) for k in case.key_names)
        assert case.key_right == tuple(tuple(expected[k][1]) for k in case.key_names)
    # Fixed predicate/name choices, not chosen for outcomes. Only one exact
    # scan per dataset: validates source mapping and actual full-reference SQL.
    smoke = []
    selected = (("dbpedia", "agent", "single"), ("goemotions", "sent_positive", "multi"))
    for dataset, left, variant in selected:
        case = next(c for c in cases if c.dataset == dataset and c.predicate_left == f"{dataset}/{left}" and c.variant == variant)
        result = run_exact(case, "exact_scan")
        assert result["output"]["exact"] and result["helper_requests"] == 0
        smoke.append({"case_id": case.case_id, "oracle_requests": result["oracle_requests"],
            "helper_requests": result["helper_requests"], "full_count": result["output"]["full_count"],
            "returned_count": result["output"]["returned_count"], "exact": result["output"]["exact"],
            "sql_timings": {k: v for k, v in result["output"].items() if k.endswith("seconds")}})
        case.close_sql()
    return {"metadata_cases_checked": 32, "metadata_loading_seconds": metadata_seconds,
            "scope": "metadata for all 32; two predetermined exact-scan SQL smoke checks only; no study-arm matrix", "smoke": smoke}


def main():
    started = time.perf_counter()
    result = {"purpose": "implementation correctness only, not performance study or coverage experiment",
              "tiny": tiny_checks(), "production": production_metadata_and_smoke()}
    result["elapsed_seconds"] = time.perf_counter() - started
    output = HERE / "tests"; output.mkdir(exist_ok=True)
    (output / "harness_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
