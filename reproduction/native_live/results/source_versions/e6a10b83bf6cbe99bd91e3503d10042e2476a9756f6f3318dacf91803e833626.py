"""Strong no-audit exact controls for the locked external join study.

Each policy owns an Oracle cache. All output facts are validated by actual
DuckDB returned-derivation materialization; no retrospective minimum is used.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from heapq import heappop, heappush
from typing import Mapping
import time

import numpy as np

try:
    from .common import Case, Oracle, evaluate, make_join, possible_unknowns
except ImportError:
    from common import Case, Oracle, evaluate, make_join, possible_unknowns

METHODS = ("exact_scan", "left_first", "right_first", "cover", "helper_greedy")


class LivePartners:
    """Incremental factorized possible-partner counts; reads only paid facts.

    Deleting a negative changes each incident biclique once. A pair repeated
    across keys contributes per key. Known positives stay live, unknowns with
    zero possible degree need no test. Optional score weights stay fixed so
    every weighted degree only decreases, permitting a lazy maximum heap.
    """
    def __init__(self, case: Case, facts: Mapping[int, int], scores=None):
        self.case = case
        self.scores = np.ones(case.n) if scores is None else np.asarray(scores, dtype=np.float64)
        self.known = np.zeros(case.n, dtype=bool)
        self.alive = np.ones(case.n, dtype=bool)
        for j, y in facts.items():
            self.known[j] = True
            if y == 0: self.alive[j] = False
        self.incident = [[] for _ in range(case.n)]
        self.degree = np.zeros(case.n, dtype=np.int64)
        self.weighted_degree = np.zeros(case.n, dtype=np.float64)
        for k, (left, right) in enumerate(zip(case.key_left, case.key_right)):
            nl = sum(bool(self.alive[j]) for j in left)
            nr = sum(bool(self.alive[j]) for j in right)
            sl = sum(float(self.scores[j]) for j in left if self.alive[j])
            sr = sum(float(self.scores[j]) for j in right if self.alive[j])
            for ids, count, weight in ((left, nr, sr), (right, nl, sl)):
                for j in ids:
                    self.incident[j].append(k)
                    if self.alive[j]:
                        self.degree[j] += count; self.weighted_degree[j] += weight
        self.neighbor_updates = 0
        self.heap_pops = 0

    def eligible(self, j: int) -> bool:
        return bool(not self.known[j] and self.alive[j] and self.degree[j] > 0)

    def observe(self, new: Mapping[int, int]) -> None:
        for j, y in new.items():
            if self.known[j]: raise ValueError("Live graph received duplicate label")
            self.known[j] = True
            if y:
                continue
            self.alive[j] = False
            self.degree[j] = 0
            self.weighted_degree[j] = 0.
            for k in self.incident[j]:
                opposite = self.case.key_right[k] if self.case.side[j] == 0 else self.case.key_left[k]
                for u in opposite:
                    if self.alive[u]:
                        self.degree[u] -= 1
                        self.weighted_degree[u] = max(0., self.weighted_degree[u] - self.scores[j])
                        self.neighbor_updates += 1
                        if self.degree[u] < 0: raise AssertionError("Negative possible degree")

    def priority(self, j: int) -> float:
        return float(self.scores[j] * self.weighted_degree[j])


def _jsonable(value):
    if isinstance(value, Fraction): return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)): return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.generic): return value.item()
    return value


def run_exact(case: Case, method: str, *, initial_facts=None, materialize=True) -> dict:
    aliases = {"scan": "exact_scan", "full_scan": "exact_scan", "left": "left_first", "right": "right_first", "mincut": "cover", "greedy": "helper_greedy"}
    method = aliases.get(method, method)
    if method not in METHODS: raise ValueError(f"Unknown exact control {method}")
    oracle = Oracle(case, initial_facts)
    timing = {"preprocessing_seconds": 0., "graph_cut_seconds": 0., "helper_access_seconds": 0., "execution_bookkeeping_seconds": 0.}
    diagnostics = {}
    started = time.perf_counter()
    if method == "exact_scan":
        # Structural zero-fan-out elimination only; full scan does not depend
        # on the observed negatives of other newly evaluated identities.
        required = possible_unknowns(case, {})
        timing["preprocessing_seconds"] = time.perf_counter() - started
        started = time.perf_counter(); oracle.acquire(required, "execution")
        timing["execution_bookkeeping_seconds"] = time.perf_counter() - started
    elif method in ("left_first", "right_first"):
        live = LivePartners(case, oracle.facts)
        timing["preprocessing_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        first = 0 if method == "left_first" else 1
        for side in (first, 1 - first):
            for j in range(case.n):
                if case.side[j] == side and live.eligible(j):
                    live.observe(oracle.acquire((j,), "execution"))
        timing["execution_bookkeeping_seconds"] = time.perf_counter() - started
        diagnostics = {"neighbor_updates": live.neighbor_updates}
    elif method == "cover":
        join = make_join(case)
        timing["preprocessing_seconds"] = time.perf_counter() - started
        started = time.perf_counter(); plan = join.plan_cut(oracle.facts)
        verification = join.verify_cut(plan, oracle.facts)
        timing["graph_cut_seconds"] = time.perf_counter() - started
        started = time.perf_counter(); oracle.acquire(plan.acquire, "cut")
        residual = join.linearize(oracle.facts)
        required = tuple(j for j, w in residual.weights if w > 0)
        oracle.acquire(required, "execution")
        timing["execution_bookkeeping_seconds"] = time.perf_counter() - started
        diagnostics = {"cut_acquire": list(plan.acquire), "cut_cost": _jsonable(plan.cost),
            "cut_verification": _jsonable(asdict(verification)), "cut_certificate": _jsonable(asdict(plan.certificate)),
            "residual_constant": residual.constant, "residual_weights": [[int(j), int(w)] for j, w in residual.weights]}
    else:
        started = time.perf_counter(); scores = oracle.use_scores(range(case.n))
        timing["helper_access_seconds"] = time.perf_counter() - started
        started = time.perf_counter(); live = LivePartners(case, oracle.facts, scores)
        heap = [(-live.priority(j), j) for j in range(case.n) if live.eligible(j)]
        from heapq import heapify
        heapify(heap)
        timing["preprocessing_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        while heap:
            negative_old, j = heappop(heap); live.heap_pops += 1
            if not live.eligible(j): continue
            priority = live.priority(j)
            # All other heap keys remain upper bounds on current priorities.
            if priority != -negative_old:
                heappush(heap, (-priority, j)); continue
            live.observe(oracle.acquire((j,), "execution"))
        timing["execution_bookkeeping_seconds"] = time.perf_counter() - started
        diagnostics = {"neighbor_updates": live.neighbor_updates, "heap_pops": live.heap_pops,
            "ranking": "fixed helper score times dynamic sum of possible-partner helper scores per shared key"}
    if possible_unknowns(case, oracle.facts):
        raise AssertionError("An exact policy left an unresolved possible positive derivation")
    result = {"case_id": case.case_id, "method": method, "no_audit": True,
        **oracle.summary(), "facts": dict(oracle.facts), "ledger": list(oracle.ledger),
        "helper_ledger": list(oracle.helper_ledger), "timings": timing, "diagnostics": diagnostics,
        "target_version": case.target_version, "oracle_version": case.oracle_version}
    if materialize:
        result["output"] = evaluate(case, oracle.facts)
        if not result["output"]["exact"]: raise AssertionError("Exact control differs from full-tape materialized SQL")
    return result


def run_all_exact(case: Case, *, initial_facts=None, materialize=True) -> list[dict]:
    return [run_exact(case, method, initial_facts=initial_facts, materialize=materialize) for method in METHODS]
