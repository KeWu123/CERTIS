"""Post-matrix stronger exact controls; not retroactively preregistered.

Both complete executors select from the current possible-partner graph, use
the Oracle's function cache, and eliminate unknown identities only when their
current possible degree is zero. The negative-first policy uses a fixed
helper estimate of a negative label; it makes no probabilistic exactness claim.
All helper score access is charged through Oracle.use_scores. Actual output
materialization and independent full-tape validation belong to common.evaluate.
"""
from __future__ import annotations

from heapq import heapify, heappop, heappush
import time

import numpy as np

try:
    from .common import Case, Oracle, evaluate, possible_unknowns
    from .exact_baselines import LivePartners
except ImportError:
    from common import Case, Oracle, evaluate, possible_unknowns
    from exact_baselines import LivePartners


STRATEGIES = ("dynamic_max_degree", "helper_negative_first")
METHODS = STRATEGIES
REGISTRATION_NOTE = (
    "Post-matrix stronger controls added after the original matrix; "
    "not retroactively preregistered and not part of the locked original arms."
)


def run_extra_exact(case: Case, strategy: str, materialize=True, *, initial_facts=None) -> dict:
    """Run a complete post-matrix exact executor with unit identity costs.

    dynamic_max_degree chooses the unknown with greatest current possible
    degree without a helper. helper_negative_first maximizes
    (1 - score_i) * current_possible_degree_i. A pair sharing multiple keys
    contributes once per key. Ties use the stable integer identity order.
    Scores are conservatively purchased for the complete target, as in the
    original helper-greedy exact control. Selection never reads tape labels;
    every new fact is obtained by Oracle.acquire.
    """
    aliases = {
        "max_degree": "dynamic_max_degree",
        "metadata_max_degree": "dynamic_max_degree",
        "degree_greedy": "dynamic_max_degree",
        "negative_first": "helper_negative_first",
        "negative_first_greedy": "helper_negative_first",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown post-matrix exact control {strategy}")
    oracle = Oracle(case, initial_facts)
    timing = {
        "preprocessing_seconds": 0., "graph_cut_seconds": 0.,
        "helper_access_seconds": 0., "execution_bookkeeping_seconds": 0.,
    }
    started = time.perf_counter()
    live = LivePartners(case, oracle.facts)
    timing["preprocessing_seconds"] = time.perf_counter() - started
    if strategy == "helper_negative_first":
        started = time.perf_counter()
        weight = 1. - oracle.use_scores(range(case.n))
        timing["helper_access_seconds"] = time.perf_counter() - started
    else:
        weight = np.ones(case.n, dtype=np.float64)

    def priority(j: int) -> float:
        # The degree is an exact metadata count, not a sum of helper scores.
        return float(weight[j] * int(live.degree[j]))

    started = time.perf_counter()
    heap = [(-priority(j), j) for j in range(case.n) if live.eligible(j)]
    heapify(heap)
    timing["preprocessing_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    while heap:
        negative_old, j = heappop(heap)
        live.heap_pops += 1
        if not live.eligible(j):
            continue
        current = priority(j)
        # Only paid negative observations remove possible partners. Thus each
        # stale key is an upper bound, including ties and zero helper weights.
        if current != -negative_old:
            heappush(heap, (-current, j))
            continue
        live.observe(oracle.acquire((j,), "execution"))
    timing["execution_bookkeeping_seconds"] = time.perf_counter() - started
    if possible_unknowns(case, oracle.facts):
        raise AssertionError("Post-matrix exact executor left an unresolved possible derivation")
    ranking = (
        "dynamic current possible degree; unit identity cost; no helper"
        if strategy == "dynamic_max_degree" else
        "(1 - fixed helper score_i) * dynamic current possible degree_i; unit identity cost"
    )
    result = {
        "case_id": case.case_id, "method": strategy, "no_audit": True,
        **oracle.summary(), "facts": dict(oracle.facts),
        "ledger": list(oracle.ledger), "helper_ledger": list(oracle.helper_ledger),
        "timings": timing,
        "diagnostics": {
            "neighbor_updates": live.neighbor_updates, "heap_pops": live.heap_pops,
            "ranking": ranking, "tie_break": "ascending integer identity",
            "helper_charge_scope": "all target identities once" if strategy == "helper_negative_first" else "none",
        },
        "post_matrix_control": True, "preregistered": False,
        "registration_note": REGISTRATION_NOTE,
        "target_version": case.target_version, "oracle_version": case.oracle_version,
    }
    if materialize:
        result["output"] = evaluate(case, oracle.facts)
        if not result["output"]["exact"]:
            raise AssertionError("Post-matrix exact control differs from materialized full-tape SQL")
    return result


def run_all_extra_exact(case: Case, *, initial_facts=None, materialize=True) -> list[dict]:
    return [run_extra_exact(case, strategy, materialize, initial_facts=initial_facts)
            for strategy in STRATEGIES]
