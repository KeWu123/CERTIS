"""Post-matrix exploratory mechanism controls; not retroactive preregistration.

These cold-cache, no-helper policies retain the original direct degree order,
audit, 65 prefix masks, batch size, and exact dynamic partner pruning. They
change only the specified certificate objective or whether its mask advances.
No algorithm receives validation labels; all facts come from Oracle.acquire.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Oracle, evaluate, possible_unknowns, potential_degrees
from direct_arms import (ALLOWANCE, DELTA, Envelope, InconsistentEvidence,
                         RegionSide, key_support, make_regions)

METHODS = ("keywise_degree", "fixed_mask_pruning_degree", "separate_extrema_degree")
REGISTRATION_NOTE = (
    "Post-matrix exploratory mechanism controls; not original preregistered arms "
    "and not retroactively preregistered."
)


def separate_extrema_components(keys, allowance=ALLOWANCE):
    """Safe integer q*max(lost)-p*min(total), separating both extrema.

    Each key's lost maximum uses the existing exact polygon kernel at zero
    allowance. This also checks both domains before a lower total can be used.
    Shared identities across keys retain the original safe relaxation.
    """
    if not isinstance(allowance, Fraction) or not 0 <= allowance <= 1:
        raise ValueError("A Fraction allowance in [0,1] is required")
    lost_max = total_min = key_count = 0
    for left, right in keys:
        lost_max += key_support(left, right, Fraction(0))
        minima = []
        for side in (left, right):
            lo = max(side.lower_total, side.known_positive)
            hi = min(side.upper_total, side.known_positive + side.unknown_lost + side.unknown_kept)
            if lo > hi:
                raise InconsistentEvidence("Empty clipped count domain in separate-extrema control")
            minima.append(lo)
        total_min += minima[0] * minima[1]
        key_count += 1
    return {"scaled_support": allowance.denominator * lost_max - allowance.numerator * total_min,
            "lost_max": lost_max, "total_min": total_min, "key_count": key_count}


class SeparateExtremaEnvelope(Envelope):
    def support(self, facts, cardinality):
        self.calls += 1
        labels = np.full(self.case.n, -1, dtype=np.int8)
        for j, value in facts.items():
            labels[j] = value
        keys = []
        for pair, original in zip(self.groups, zip(self.case.key_left, self.case.key_right)):
            sides = []
            for ids, original_ids in zip(pair, original):
                known = labels[ids]
                unknown = known < 0
                lost = int(np.count_nonzero(unknown & (self.position[ids] < cardinality)))
                kept = int(np.count_nonzero(unknown)) - lost
                positive = int(np.count_nonzero(known == 1))
                lo, hi = self.intervals.get(original_ids, (0, 0))
                sides.append(RegionSide(lo, hi, positive, lost, kept))
            keys.append(tuple(sides))
        self.last_components = separate_extrema_components(keys)
        return self.last_components["scaled_support"]


def _snapshot(case, oracle, order, index, cardinalities, support, conflict):
    base_discarded = {int(j) for j in order[:cardinalities[index]]}
    discarded_unknown = base_discarded - oracle.facts.keys()
    retained_unknown = set(range(case.n)) - base_discarded - oracle.facts.keys()
    retained = np.ones(case.n, dtype=bool)
    retained[list(base_discarded)] = False
    return {"mask_index": index, "discard_cardinality": cardinalities[index],
            "certified_discard_cardinality": cardinalities[index] if support is not None and support <= 0 and not conflict else None,
            "scaled_support": support, "conflict_freeze": conflict,
            "fact_count": len(oracle.facts),
            "known_positive_count": sum(oracle.facts.values()),
            "known_negative_count": len(oracle.facts) - sum(oracle.facts.values()),
            "discarded_unknown_count": len(discarded_unknown),
            "retained_unknown_count": len(retained_unknown),
            "possible_retained_unknown_count": len(possible_unknowns(case, oracle.facts, retained))}


def run_mechanism(case, method: str, seed: int, audit_budget: int, *,
                  materialize=True, initial_facts=None):
    """Run one exploratory mechanism control with run_direct-compatible output."""
    if method not in METHODS:
        raise ValueError(f"Unknown exploratory mechanism control {method}")
    started = time.perf_counter()
    oracle = Oracle(case, initial_facts)
    if initial_facts:
        raise ValueError("Matched mechanism controls require the original cold target cache")
    prep = time.perf_counter()
    scores = np.ones(case.n, dtype=np.float64)
    rank = scores * potential_degrees(case, {}, opposite_scores=scores)
    order = np.lexsort((np.arange(case.n), rank))
    execution_order = np.lexsort((np.arange(case.n), -rank))
    cardinalities = [case.n * step // 64 for step in range(65)]
    preprocessing = time.perf_counter() - prep
    intervals, regions, audit_seconds = make_regions(case, oracle, seed, audit_budget)
    if method == "separate_extrema_degree":
        envelope = SeparateExtremaEnvelope(case, intervals, order)
    else:
        envelope = Envelope(case, intervals, order, keywise=method == "keywise_degree")
    advance_mask = method != "fixed_mask_pruning_degree"
    conflict = False
    conflicts = []
    projection_started = time.perf_counter()
    try:
        index = envelope.advance(oracle.facts, 0, cardinalities)
        initial_support = envelope.support(oracle.facts, cardinalities[index])
    except InconsistentEvidence as exc:
        # With no valid initial domain, freeze the no-discard mask and execute
        # its exact retained query. An inconsistent domain is never certified.
        index, initial_support, conflict = 0, None, True
        conflicts.append({"stage": "initial_projection", "paid": len(oracle.ledger), "error": str(exc)})
    projection_seconds = time.perf_counter() - projection_started
    initial_index = index
    initial_retained = set(range(case.n)) - set(int(j) for j in order[:cardinalities[index]]) - oracle.facts.keys()
    initial_snapshot = _snapshot(case, oracle, order, index, cardinalities, initial_support, conflict)
    initial_facts_after_audit = dict(oracle.facts)
    events = [{"paid": len(oracle.ledger), "mask_index": index,
               "discard_cardinality": cardinalities[index], "conflict": conflict}]
    bookkeeping_seconds = 0.
    while True:
        bookkeeping_started = time.perf_counter()
        retained = np.ones(case.n, dtype=bool)
        retained[order[:cardinalities[index]]] = False
        possible = set(possible_unknowns(case, oracle.facts, retained))
        selected = [int(j) for j in execution_order if int(j) in possible][:128]
        bookkeeping_seconds += time.perf_counter() - bookkeeping_started
        if not selected:
            break
        oracle.acquire(selected, "execution")
        if not conflict:
            projection_started = time.perf_counter()
            try:
                if advance_mask:
                    index = envelope.advance(oracle.facts, index, cardinalities)
                else:
                    # The mask stays fixed, while paid facts still detect an
                    # inconsistent domain and update exact partner pruning.
                    envelope.support(oracle.facts, cardinalities[index])
            except InconsistentEvidence as exc:
                conflict = True
                conflicts.append({"stage": "execution_feedback", "paid": len(oracle.ledger), "error": str(exc)})
            projection_seconds += time.perf_counter() - projection_started
        events.append({"paid": len(oracle.ledger), "mask_index": index,
                       "discard_cardinality": cardinalities[index], "conflict": conflict})
    algorithm_seconds = time.perf_counter() - started
    discarded = [int(j) for j in order[:cardinalities[index]] if int(j) not in oracle.facts]
    retained = np.ones(case.n, dtype=bool)
    retained[discarded] = False
    if possible_unknowns(case, oracle.facts, retained):
        raise AssertionError("Mechanism control left an unresolved retained derivation")
    execution_ids = {row["id"] for row in oracle.ledger if row["phase"] == "execution"}
    if not execution_ids <= initial_retained:
        raise AssertionError("Mechanism control violated initial retained request containment")
    if not advance_mask and (index != initial_index or any(event["mask_index"] != initial_index for event in events)):
        raise AssertionError("Fixed-mask control advanced its initial mask")
    output = evaluate(case, oracle.facts) if materialize else None
    try:
        final_support = envelope.support(oracle.facts, cardinalities[index])
    except InconsistentEvidence as exc:
        final_support = None
        if not conflict:
            conflict = True
            conflicts.append({"stage": "final_check", "paid": len(oracle.ledger), "error": str(exc)})
    final_snapshot = _snapshot(case, oracle, order, index, cardinalities, final_support, conflict)
    return {"method": method, "seed": seed, "audit_budget": audit_budget,
            "mode": envelope.mode, "allowance": str(ALLOWANCE), "delta": str(DELTA),
            "certificate_target": "each key separately" if method == "keywise_degree" else "whole answer bag",
            "certificate_objective": "separate lost maximum and total minimum" if method == "separate_extrema_degree" else "original signed coupled support",
            "mask_feedback": advance_mask, "dynamic_partner_pruning": True,
            "summary": oracle.summary(), "oracle_ledger": list(oracle.ledger),
            "helper_ledger": list(oracle.helper_ledger), "facts": dict(oracle.facts),
            "regions": regions, "discarded": discarded,
            "initial_mask_index": initial_index, "final_mask_index": index,
            "initial_discard_cardinality": cardinalities[initial_index],
            "final_discard_cardinality": cardinalities[index],
            "initial_certified_cardinality": initial_snapshot["certified_discard_cardinality"],
            "final_certified_cardinality": final_snapshot["certified_discard_cardinality"],
            "initial_retained_unknown_count": len(initial_retained),
            "initial_retained_unknown": sorted(initial_retained), "initial_R0": sorted(initial_retained),
            "initial_facts_after_audit": initial_facts_after_audit,
            "initial_certificate": initial_snapshot, "final_certificate": final_snapshot,
            "request_containment_checked": True, "conflict_freeze": conflict,
            "conflicts": conflicts, "initial_scaled_support": initial_support,
            "final_scaled_support": final_support, "projection_calls": envelope.calls,
            "events": events, "output": output,
            "post_matrix_control": True, "exploratory": True, "preregistered": False,
            "registration_note": REGISTRATION_NOTE,
            "ranking": "original direct degree ranking, scores=1, no helper",
            "mask_family_size": 65, "execution_batch_size": 128,
            "timings": {"preprocessing_seconds": preprocessing, "audit_seconds": audit_seconds,
                        "projection_seconds": projection_seconds,
                        "execution_bookkeeping_seconds": bookkeeping_seconds,
                        "algorithm_seconds": algorithm_seconds}}


run_mechanism_control = run_mechanism
