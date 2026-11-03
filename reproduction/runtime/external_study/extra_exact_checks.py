"""Tiny checks for post-matrix stronger controls, not retroactive preregistration.

This checks synthetic Boolean fixtures only; it does not execute the external
study matrix, inspect its scores, or use its results to select a control.
Run this file directly to regenerate extra_exact_checks.json next to it.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import time

import numpy as np

try:
    from .common import Oracle, make_tiny_case
    from .extra_exact_controls import REGISTRATION_NOTE, STRATEGIES, run_extra_exact
except ImportError:
    from common import Oracle, make_tiny_case
    from extra_exact_controls import REGISTRATION_NOTE, STRATEGIES, run_extra_exact


def naive_recomputed_order(case, strategy, initial_facts=None):
    """Independent full metadata recomputation; observations remain charged."""
    oracle = Oracle(case, initial_facts)
    weights = 1. - oracle.use_scores(range(case.n)) if strategy == "helper_negative_first" else np.ones(case.n)
    while True:
        degree = [0] * case.n
        for left, right in zip(case.key_left, case.key_right):
            for j in left:
                for u in right:
                    if oracle.facts.get(j) != 0 and oracle.facts.get(u) != 0:
                        degree[j] += 1
                        degree[u] += 1
        candidates = [j for j in range(case.n) if j not in oracle.facts and degree[j] > 0]
        if not candidates:
            break
        j = min(candidates, key=lambda j: (-float(weights[j] * degree[j]), j))
        oracle.acquire((j,), "execution")
    return oracle


def _bag(case, labels):
    return sorted((k, j, u) for k, (left, right) in enumerate(zip(case.key_left, case.key_right))
                  for j in left for u in right if labels.get(j) == 1 and labels.get(u) == 1)


def check_case(case, truth, *, initial_facts=None, materialize=True):
    reference_bag = _bag(case, dict(enumerate(truth)))
    records = []
    for strategy in STRATEGIES:
        expected = naive_recomputed_order(case, strategy, initial_facts)
        result = run_extra_exact(case, strategy, materialize, initial_facts=initial_facts)
        actual_ids = [row["id"] for row in result["ledger"]]
        expected_ids = [row["id"] for row in expected.ledger]
        assert actual_ids == expected_ids, (strategy, actual_ids, expected_ids)
        assert result["facts"] == expected.facts
        assert result["ledger"] == expected.ledger
        assert result["helper_ledger"] == expected.helper_ledger
        assert len(actual_ids) == len(set(actual_ids))
        assert not set(actual_ids).intersection(initial_facts or {})
        assert result["initial_cache_count"] == len(initial_facts or {})
        assert result["oracle_requests"] == len(actual_ids)
        assert result["phase_counts"] == {"audit": 0, "cut": 0, "execution": len(actual_ids)}
        assert result["oracle_tokens_in"] == sum(row["tokens_in"] for row in result["ledger"])
        assert result["oracle_tokens_out"] == sum(row["tokens_out"] for row in result["ledger"])
        assert result["helper_requests"] == (case.n if strategy == "helper_negative_first" else 0)
        assert result["helper_read_items"] == (case.n if strategy == "helper_negative_first" else 0)
        assert result["helper_read_calls"] == (1 if strategy == "helper_negative_first" else 0)
        assert result["helper_tokens_in"] == sum(row["tokens_in"] for row in result["helper_ledger"])
        assert result["helper_tokens_out"] == sum(row["tokens_out"] for row in result["helper_ledger"])
        assert _bag(case, result["facts"]) == reference_bag
        assert result["post_matrix_control"] and result["preregistered"] is False
        if materialize:
            output = result["output"]
            assert output["exact"] and output["bag_subset"]
            assert output["full_count"] == output["returned_count"] == len(reference_bag)
            assert output["lost_count"] == 0
            # Compare actual materialized tuples, retaining each shared key.
            materialized = case._sql["connection"].execute(
                "SELECT key_id,left_id,right_id FROM returned_derivations ORDER BY ALL"
            ).fetchall()
            assert materialized == reference_bag
        else:
            assert "output" not in result
        records.append({"strategy": strategy, "selection_order": actual_ids,
                        "oracle_requests": result["oracle_requests"],
                        "helper_requests": result["helper_requests"],
                        "full_bag_count": len(reference_bag), "materialized": materialize})
    return records


def main():
    started = time.perf_counter()
    counts = {"boolean_cases": 0, "random_cases": 0, "cached_cases": 0,
              "policy_runs": 0, "materialized_policy_runs": 0}
    examples = []
    # Complete biclique, repeated-key multiplicity, and isolated/one-sided keys.
    fixtures = (
        (((0, 1),), ((2, 3),), (0.9, 0.1, 0.2, 1.0)),
        (((0, 1), (0,), (1,)), ((2,), (2, 3), (3,)), (0.5, 0.5, 0.5, 0.5)),
        (((0,), (1,), ()), ((2,), (), (3,)), (1., 1., 1., 1.)),
    )
    for fixture_index, (left, right, scores) in enumerate(fixtures):
        for labels in product((0, 1), repeat=4):
            case = make_tiny_case(side=(0, 0, 1, 1), key_left=left, key_right=right,
                                  truth=labels, scores=scores,
                                  name=f"post_matrix_boolean_{fixture_index}_{''.join(map(str, labels))}",
                                  tokens={"oracle_input_tokens": (2, 3, 5, 7),
                                          "oracle_output_tokens": (1, 2, 3, 4),
                                          "helper_input_tokens": (11, 13, 17, 19),
                                          "helper_output_tokens": (2, 4, 6, 8)})
            try:
                records = check_case(case, labels)
                counts["boolean_cases"] += 1
                counts["policy_runs"] += len(records)
                counts["materialized_policy_runs"] += len(records)
                if fixture_index == 0 and labels in ((0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)):
                    examples.append({"fixture": fixture_index, "labels": labels, "runs": records})
                if labels in ((0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)):
                    records = check_case(case, labels, initial_facts={0: labels[0], 2: labels[2]})
                    counts["cached_cases"] += 1
                    counts["policy_runs"] += len(records)
                    counts["materialized_policy_runs"] += len(records)
            finally:
                case.close_sql()
    rng = np.random.default_rng(20261016)
    for index in range(12):
        nl, nr = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        n = nl + nr
        keys = int(rng.integers(1, 8))
        left = tuple(tuple(j for j in range(nl) if rng.random() < .55) for _ in range(keys))
        right = tuple(tuple(j for j in range(nl, n) if rng.random() < .55) for _ in range(keys))
        labels = rng.integers(0, 2, size=n).tolist()
        scores = rng.choice((0., .1, .5, .9, 1.), size=n)
        case = make_tiny_case(side=[0] * nl + [1] * nr, key_left=left, key_right=right,
                              truth=labels, scores=scores, name=f"post_matrix_random_{index}")
        try:
            # No SQL for these additional selection-order checks.
            records = check_case(case, labels, initial_facts={0: labels[0]}, materialize=False)
            counts["random_cases"] += 1
            counts["policy_runs"] += len(records)
        finally:
            case.close_sql()
    report = {
        "status": "passed", "post_matrix_control": True, "preregistered": False,
        "registration_note": REGISTRATION_NOTE, "scope": "tiny synthetic correctness checks only; no external matrix execution",
        "strategies": list(STRATEGIES), "counts": counts,
        "checks": ["all Boolean assignments on three four-identity graphs",
                   "actual DuckDB bag materialization equals explicit reference tuples",
                   "lazy incremental selection equals naive full metadata recomputation at each step",
                   "function cache excludes repeat purchases and preserves initial paid facts",
                   "zero-partner pruning, overlapping-key bag multiplicity, empty outputs",
                   "stable identity ties, zero negative-first weights, score endpoints",
                   "helper score access and distinct identity/token accounting"],
        "examples": examples, "elapsed_seconds": time.perf_counter() - started,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts, "artifact": str(path)}))


if __name__ == "__main__":
    main()
