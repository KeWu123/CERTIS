"""Direct bilinear certificate arms for the protocol-locked external study.

Only Oracle.acquire reads target truth. Masks/ranking/evidence domains are
constructed from metadata, paid helper values and already acquired facts.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import math
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from bilinear_certificate import RegionSide, InconsistentEvidence, key_support
from exact_hypergeom import count_interval_exact
from common import Oracle, evaluate, potential_degrees, possible_unknowns

ALLOWANCE = Fraction(1, 10)
DELTA = Fraction(1, 20)


def downward_alpha(nominal: Fraction) -> float:
    value = float(nominal)
    if Fraction.from_float(value) > nominal:
        value = math.nextafter(value, 0.)
    if not 0 < value < 1 or Fraction.from_float(value) > nominal:
        raise ValueError("Cannot represent a positive covered numeric budget")
    return value


def make_regions(case, oracle, seed: int, budget: int):
    """One SRS per side, with conditional HG inversion for each fixed region."""
    started = time.perf_counter()
    if budget < 0:
        raise ValueError("Nonnegative audit budget required")
    rng = np.random.default_rng(seed)
    sampled = []
    for side in (0, 1):
        population = np.flatnonzero(case.side == side)
        drawn = rng.choice(population, size=min(budget, len(population)), replace=False)
        sampled.extend(int(j) for j in drawn)
    oracle.acquire(sampled, "audit")
    sample_set = set(sampled)
    # Empty side/keys need no statistical record. Identical subsets share one.
    domains = tuple(dict.fromkeys(tuple(ids) for pair in zip(case.key_left, case.key_right)
                                  for ids in pair if ids))
    alpha_nominal = DELTA / max(1, len(domains))
    alpha = downward_alpha(alpha_nominal)
    records, intervals = [], {}
    for ids in domains:
        n = sum(j in sample_set for j in ids)
        k = sum(oracle.facts[j] for j in ids if j in sample_set)
        lo, hi = count_interval_exact(len(ids), n, k, alpha)
        intervals[ids] = (int(lo), int(hi))
        records.append({"ids": list(ids), "population": len(ids), "sample": n,
                        "positive_sample": k, "lower": int(lo), "upper": int(hi),
                        "alpha_nominal": str(alpha_nominal),
                        "alpha_executed": str(Fraction.from_float(alpha)),
                        "alpha_float_hex": alpha.hex()})
    return intervals, records, time.perf_counter() - started


class Envelope:
    def __init__(self, case, intervals, order, *, keywise=False):
        self.case, self.intervals = case, intervals
        self.position = np.empty(case.n, dtype=np.int64)
        self.position[order] = np.arange(case.n)
        self.groups = tuple(tuple(np.asarray(ids, dtype=np.int64) for ids in pair)
                            for pair in zip(case.key_left, case.key_right))
        membership = np.zeros(case.n, dtype=np.int64)
        for a, b in self.groups:
            membership[a] += 1; membership[b] += 1
        self.mode = "single_key_exact" if np.all(membership <= 1) else "repeated_identity_relaxation"
        self.calls = 0
        self.keywise = keywise

    def support(self, facts, cardinality):
        self.calls += 1
        labels = np.full(self.case.n, -1, dtype=np.int8)
        for j, y in facts.items(): labels[j] = y
        local_supports = []
        for pair, original in zip(self.groups, zip(self.case.key_left, self.case.key_right)):
            sides = []
            for ids, original_ids in zip(pair, original):
                known = labels[ids]
                unknown = known < 0
                lost = int(np.count_nonzero(unknown & (self.position[ids] < cardinality)))
                kept = int(np.count_nonzero(unknown)) - lost
                h = int(np.count_nonzero(known == 1))
                lo, hi = self.intervals.get(original_ids, (0, 0))
                sides.append(RegionSide(lo, hi, h, lost, kept))
            local_supports.append(key_support(*sides, ALLOWANCE))
        return max(local_supports, default=0) if self.keywise else sum(local_supports)

    def advance(self, facts, current, cardinalities):
        """Nested supports are monotone; never revert or silently accept emptiness."""
        self.support(facts, cardinalities[current])  # detect conflicts even at last mask
        lo, hi = current, len(cardinalities) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.support(facts, cardinalities[mid]) <= 0: lo = mid
            else: hi = mid - 1
        return lo


def run_direct(case, seed: int, audit_budget: int, *, audit=True,
               feedback=True, helper=True, materialize=True, initial_facts=None,
               keywise=False):
    """Run a complete query, retaining signed supports, phase costs and failures."""
    started = time.perf_counter()
    oracle = Oracle(case, initial_facts)
    if initial_facts:
        raise ValueError("This locked study audits cold targets; warm sources need a separate protocol")
    prep = time.perf_counter()
    scores = oracle.use_scores(range(case.n)) if helper else np.ones(case.n, dtype=np.float64)
    rank = scores * potential_degrees(case, {}, opposite_scores=scores)
    order = np.lexsort((np.arange(case.n), rank))
    execution_order = np.lexsort((np.arange(case.n), -rank))
    cardinalities = [case.n * t // 64 for t in range(65)]
    preprocessing = time.perf_counter() - prep
    intervals, records, audit_seconds = make_regions(case, oracle, seed, audit_budget if audit else 0)
    envelope = Envelope(case, intervals, order, keywise=keywise)
    t = time.perf_counter()
    index = envelope.advance(oracle.facts, 0, cardinalities)
    projection_seconds = time.perf_counter() - t
    initial_index = index
    initial_retained = set(range(case.n)) - set(int(v) for v in order[:cardinalities[index]]) - oracle.facts.keys()
    conflict = False
    events = [{"paid": len(oracle.ledger), "mask_index": index, "discard_cardinality": cardinalities[index]}]
    bookkeeping_seconds = 0.
    while True:
        t = time.perf_counter()
        retained = np.ones(case.n, dtype=bool)
        retained[order[:cardinalities[index]]] = False
        if feedback:
            possible = set(possible_unknowns(case, oracle.facts, retained))
        else:
            possible = initial_retained - oracle.facts.keys()
        selected = [int(v) for v in execution_order if int(v) in possible][:128]
        bookkeeping_seconds += time.perf_counter() - t
        if not selected: break
        oracle.acquire(selected, "execution")
        if feedback and not conflict:
            t = time.perf_counter()
            try:
                index = envelope.advance(oracle.facts, index, cardinalities)
            except InconsistentEvidence:
                conflict = True
            projection_seconds += time.perf_counter() - t
        events.append({"paid": len(oracle.ledger), "mask_index": index,
                       "discard_cardinality": cardinalities[index], "conflict": conflict})
    algorithm_seconds = time.perf_counter() - started
    discarded = [int(v) for v in order[:cardinalities[index]] if int(v) not in oracle.facts]
    # Positive cached endpoints are automatically present even when inside the
    # base mask. All potential retained positive pairs must now be fully known.
    retained = np.ones(case.n, dtype=bool); retained[discarded] = False
    if possible_unknowns(case, oracle.facts, retained):
        raise AssertionError("Executor stopped before retained derivations were determined")
    execution_ids = {a["id"] for a in oracle.ledger if a["phase"] == "execution"}
    if not execution_ids <= initial_retained:
        raise AssertionError("Retained-only request containment failed")
    output = evaluate(case, oracle.facts) if materialize else None
    try: final_support = envelope.support(oracle.facts, cardinalities[index])
    except InconsistentEvidence: final_support = None
    return {"method": ("keywise_" if keywise else "direct_") + ("feedback" if feedback else "static") +
            ("_audit" if audit else "_point") + ("_helper" if helper else "_degree"),
            "seed": seed, "audit_budget": audit_budget if audit else 0,
            "mode": envelope.mode, "allowance": str(ALLOWANCE), "delta": str(DELTA),
            "certificate_target": "each key separately" if keywise else "whole answer bag",
            "summary": oracle.summary(), "oracle_ledger": oracle.ledger,
            "helper_ledger": oracle.helper_ledger, "facts": oracle.facts,
            "regions": records, "discarded": discarded,
            "initial_mask_index": initial_index, "final_mask_index": index,
            "initial_retained_unknown_count": len(initial_retained),
            "initial_retained_unknown": sorted(initial_retained),
            "request_containment_checked": True, "conflict_freeze": conflict,
            "final_scaled_support": final_support, "projection_calls": envelope.calls,
            "events": events, "output": output,
            "timings": {"preprocessing_seconds": preprocessing,
                        "audit_seconds": audit_seconds, "projection_seconds": projection_seconds,
                        "execution_bookkeeping_seconds": bookkeeping_seconds,
                        "algorithm_seconds": algorithm_seconds}}
