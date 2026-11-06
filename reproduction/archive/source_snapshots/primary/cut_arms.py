"""Paid structural cuts followed by fixed-target affine recall certificates.

The cut is the classical compressed bipartite vertex-cover construction in
relational_cut.py. The statistical source here is a *fresh* conditional SRS
from post-cut unknown identities, never the deterministically acquired cut.
No target truth is read except through the charged common.Oracle interface.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from pathlib import Path
import math
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / "code"))
from exact_hypergeom import count_interval_exact
from common import Oracle, evaluate, make_join

ALLOWANCE = Fraction(1, 10)
DELTA = Fraction(1, 20)
BATCH_SIZE = 128


class InconsistentEvidence(ValueError):
    """Truthful facts have emptied an original full-bin count interval."""


def downward_alpha(nominal: Fraction) -> float:
    """Return a positive binary budget covered by the nominal exact rational."""
    value = float(nominal)
    if Fraction.from_float(value) > nominal:
        value = math.nextafter(value, 0.)
    if not 0 < value < 1 or Fraction.from_float(value) > nominal:
        raise ValueError("Cannot represent a positive covered numeric budget")
    return value


def score_bins(case, scores):
    """Four full-side rank bins, fixed without any target-label access.

    Global integer identities break all score ties. Empty bins are retained
    explicitly for small fixtures. All full-side members, including cut facts,
    remain in the original region to which later facts are intersected.
    """
    groups = []
    for side in (0, 1):
        ids = np.flatnonzero(case.side == side)
        order = ids[np.lexsort((ids, scores[ids]))]
        for b in range(4):
            group = tuple(int(v) for v in order[len(order)*b//4:len(order)*(b+1)//4])
            groups.append((side, b, group))
    return tuple(groups)


def make_regions(case, oracle, bins, seed, budget, *, audit):
    """SRS per remaining side; conditional HG intervals per fixed score bin.

    A bin draw size is ancillary: condition on the number of side-wide SRS
    draws landing in that bin, then invert its exact HG distribution. Nominal
    events are fixed from the post-cut unknown domains before new labels.
    """
    started = time.perf_counter()
    before = dict(oracle.facts)
    domains = [tuple(j for j in ids if j not in before) for _, _, ids in bins]
    event_count = sum(bool(ids) for ids in domains) if audit else 0
    nominal = DELTA / event_count if event_count else None
    alpha = downward_alpha(nominal) if nominal is not None else None
    rng = np.random.default_rng(seed)
    sampled, side_draws = [], []
    for side in (0, 1):
        population = [j for j in range(case.n) if int(case.side[j]) == side and j not in before]
        size = min(budget, len(population)) if audit else 0
        drawn = [int(j) for j in rng.choice(population, size=size, replace=False)]
        sampled.extend(drawn)
        side_draws.append({"side": side, "population": len(population),
                           "sample": size, "sample_ids": drawn})
    oracle.acquire(sampled, "audit")
    sample_set = set(sampled)
    inversion_started = time.perf_counter()
    records = []
    for (side, b, ids), domain in zip(bins, domains):
        drawn = [j for j in domain if j in sample_set]
        k = sum(oracle.facts[j] for j in drawn)
        known_positive = sum(before[j] for j in ids if j in before)
        if domain and audit:
            lo, hi = count_interval_exact(len(domain), len(drawn), k, alpha)
        else:
            lo, hi = 0, len(domain)
        records.append({"side": side, "bin": b, "ids": list(ids),
                        "post_cut_unknown_ids": list(domain),
                        "population": len(domain), "sample": len(drawn),
                        "sample_ids": drawn, "positive_sample": k,
                        "cut_known_positive": known_positive,
                        "unknown_lower": int(lo), "unknown_upper": int(hi),
                        "lower": known_positive + int(lo),
                        "upper": known_positive + int(hi),
                        "alpha_nominal": str(nominal) if domain and audit else None,
                        "alpha_executed": str(Fraction.from_float(alpha)) if domain and audit else None,
                        "alpha_float_hex": alpha.hex() if domain and audit else None})
    inversion_seconds = time.perf_counter() - inversion_started
    executed_total = sum((Fraction(r["alpha_executed"]) for r in records
                          if r["alpha_executed"] is not None), Fraction(0))
    if executed_total > DELTA:
        raise AssertionError("Executed conditional-audit budget exceeds delta")
    ledger = {"source": "fresh_post_cut_srs" if audit else "point_facts_only",
              "budget_per_remaining_side": budget if audit else 0,
              "event_count": event_count, "delta_nominal": str(DELTA),
              "delta_executed": str(executed_total), "side_draws": side_draws,
              "cut_facts_are_random_samples": False}
    return records, ledger, time.perf_counter() - started, inversion_seconds


class AffineEnvelope:
    """Exact integer support over disjoint full-bin count intervals.

    After the cut Q=K0+sum(w_j*z_j), so the scaled signed loss objective is
    -K + sum(w_j*(10*1[j in G]-1)*z_j) over currently unknown identities.
    For a bin with l<=sum(z)<=u, sorting coefficients and choosing the
    positive-prefix length clipped to [l,u] is Boolean-sharp. Zero weights
    still count toward the bin's integer total. Negative maxima stay signed.
    """
    def __init__(self, n, constant, weights, regions, order):
        self.n, self.constant = int(n), int(constant)
        self.weights = {int(j): int(w) for j, w in weights.items()}
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Nonnegative affine weights required")
        self.regions = regions
        self.position = np.empty(n, dtype=np.int64)
        self.position[order] = np.arange(n)
        flat = [j for r in regions for j in r["ids"]]
        if sorted(flat) != list(range(n)):
            raise ValueError("Disjoint full bins must partition every identity")
        self.calls = 0
        self.decisions = []

    def _prepare(self, facts):
        constant = self.constant + sum(w * facts[j] for j, w in self.weights.items() if j in facts)
        groups = []
        for r in self.regions:
            unknown = [j for j in r["ids"] if j not in facts]
            known_positive = sum(facts[j] for j in r["ids"] if j in facts)
            lo = max(0, r["lower"] - known_positive)
            hi = min(len(unknown), r["upper"] - known_positive)
            if lo > hi:
                raise InconsistentEvidence(
                    f"side={r['side']} bin={r['bin']} has residual [{lo},{hi}]")
            groups.append((unknown, lo, hi))
        return constant, groups, len(facts)

    def _support(self, prepared, cardinality):
        constant, groups, paid = prepared
        value = -constant
        for unknown, lo, hi in groups:
            coefficients = sorted((self.weights.get(j, 0) *
                                   (10 * int(self.position[j] < cardinality) - 1)
                                   for j in unknown), reverse=True)
            positive = sum(c > 0 for c in coefficients)
            count = min(hi, max(lo, positive))
            value += sum(coefficients[:count])
        self.calls += 1
        self.decisions.append({"fact_count": paid, "cardinality": int(cardinality),
                               "known_constant": constant, "scaled_support": value})
        return value

    def support(self, facts, cardinality):
        return self._support(self._prepare(facts), cardinality)

    def advance(self, facts, current, cardinalities):
        prepared = self._prepare(facts)
        if self._support(prepared, cardinalities[current]) > 0:
            raise AssertionError("A previously certified mask lost validity under nonempty intersection")
        lo, hi = current, len(cardinalities) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._support(prepared, cardinalities[mid]) <= 0:
                lo = mid
            else:
                hi = mid - 1
        return lo


def _cut_record(plan, verification):
    c = plan.certificate
    return {"acquire": [int(j) for j in plan.acquire], "cost": str(plan.cost),
            "target_version": plan.target_version, "oracle_version": plan.oracle_version,
            "verification": {"node_count": verification.node_count,
                             "arc_count": verification.arc_count,
                             "structural_arcs": verification.structural_arcs,
                             "zero_primal_dual_gap": verification.zero_primal_dual_gap},
            "certificate": {"scale": c.scale, "nodes": list(c.nodes),
                            "arcs": [{"tail": a.tail, "head": a.head,
                                      "capacity": a.capacity, "structural": a.structural}
                                     for a in c.arcs],
                            "flows": list(c.flows), "source_side": sorted(c.source_side),
                            "value": str(c.value)}}


def run_cut(case, seed: int, audit_budget: int, *, audit=True, feedback=True):
    """Complete the paid-cut arm, including actual returned-bag materialization.

    Static and feedback variants share cut facts, fresh audit, and initial
    mask. Static buys every initially retained positive-weight unknown;
    feedback requests a subset of that same residual identity set. The cut
    and audit acquisitions are charged separately and included in total cost.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    if isinstance(audit_budget, bool) or not isinstance(audit_budget, Integral) or audit_budget < 0:
        raise ValueError("Nonnegative integer per-remaining-side audit budget required")
    if not isinstance(audit, bool) or not isinstance(feedback, bool):
        raise ValueError("audit and feedback must be bool")
    seed, audit_budget = int(seed), int(audit_budget)
    started = time.perf_counter()
    oracle = Oracle(case)
    t = time.perf_counter()
    scores = oracle.use_scores(range(case.n))  # Conservative complete helper charge.
    bins = score_bins(case, scores)
    join = make_join(case)
    preprocessing_seconds = time.perf_counter() - t
    t = time.perf_counter()
    plan = join.plan_cut(oracle.facts)
    verification = join.verify_cut(plan, oracle.facts)
    graph_cut_seconds = time.perf_counter() - t
    t = time.perf_counter()
    oracle.acquire(plan.acquire, "cut")
    cut_facts = dict(oracle.facts)
    acquisition_seconds = time.perf_counter() - t
    t = time.perf_counter()
    residual = join.linearize(oracle.facts)
    weights = residual.weight_map()
    rank = np.asarray([weights.get(j, 0) * float(scores[j]) for j in range(case.n)])
    order = np.lexsort((np.arange(case.n), rank))
    execution_order = np.lexsort((np.arange(case.n), -rank))
    cardinalities = [case.n * step // 64 for step in range(65)]
    affine_seconds = time.perf_counter() - t
    regions, confidence, audit_seconds, inversion_seconds = make_regions(
        case, oracle, bins, seed, audit_budget, audit=audit)
    envelope = AffineEnvelope(case.n, residual.constant, weights, regions, order)
    t = time.perf_counter()
    index = envelope.advance(oracle.facts, 0, cardinalities)
    projection_seconds = time.perf_counter() - t
    initial_index = index
    initial_retained = {j for j, w in weights.items() if w > 0 and
                        j not in oracle.facts and envelope.position[j] >= cardinalities[index]}
    events = [{"paid": len(oracle.ledger), "mask_index": index,
               "discard_cardinality": cardinalities[index]}]
    bookkeeping_seconds, conflict, conflict_reason = 0., False, None
    while True:
        t = time.perf_counter()
        selected = [int(j) for j in execution_order if j in initial_retained and
                    j not in oracle.facts and (not feedback or
                    envelope.position[j] >= cardinalities[index])][:BATCH_SIZE]
        bookkeeping_seconds += time.perf_counter() - t
        if not selected:
            break
        t = time.perf_counter()
        oracle.acquire(selected, "execution")
        acquisition_seconds += time.perf_counter() - t
        if feedback and not conflict:
            t = time.perf_counter()
            try:
                index = envelope.advance(oracle.facts, index, cardinalities)
            except InconsistentEvidence as exc:
                conflict, conflict_reason = True, str(exc)
            projection_seconds += time.perf_counter() - t
        events.append({"paid": len(oracle.ledger), "mask_index": index,
                       "discard_cardinality": cardinalities[index],
                       "execution_ids": selected, "conflict": conflict})
    discarded = [int(j) for j in order[:cardinalities[index]] if int(j) not in oracle.facts]
    if any(w > 0 and j not in oracle.facts and j not in discarded for j, w in weights.items()):
        raise AssertionError("A retained positive-weight residual identity was left unknown")
    execution_ids = {r["id"] for r in oracle.ledger if r["phase"] == "execution"}
    if not execution_ids <= initial_retained:
        raise AssertionError("Residual retained-only request containment failed")
    t = time.perf_counter()
    core_count = join.count_output(oracle.facts, discard=discarded)
    final_constant = residual.constant + sum(w * oracle.facts[j] for j, w in weights.items()
                                              if j in oracle.facts)
    if core_count != final_constant:
        raise AssertionError("Affine returned count differs from factorized output count")
    try:
        final_support = envelope.support(oracle.facts, cardinalities[index])
    except InconsistentEvidence as exc:
        final_support = None
        conflict, conflict_reason = True, str(exc)
    projection_seconds += time.perf_counter() - t
    algorithm_seconds = time.perf_counter() - started
    materialization_started = time.perf_counter()
    output = evaluate(case, oracle.facts)
    materialization_seconds = time.perf_counter() - materialization_started
    if output["returned_count"] != core_count:
        raise AssertionError("Independently materialized SQL bag differs from affine count")
    return {"method": "cut_" + ("feedback" if feedback else "static") +
            ("_audit" if audit else "_point"), "seed": seed,
            "audit_budget": audit_budget if audit else 0,
            "audit_budget_scope": "per remaining side, before fresh draw",
            "mode": "exact_affine_after_paid_cut", "allowance": str(ALLOWANCE),
            "delta": str(DELTA), "summary": oracle.summary(),
            "oracle_ledger": oracle.ledger, "helper_ledger": oracle.helper_ledger,
            "facts": dict(oracle.facts), "cut": _cut_record(plan, verification),
            "cut_facts": cut_facts, "affine_constant": residual.constant,
            "affine_weights": weights, "affine_returned_count": final_constant,
            "rank_order": [int(j) for j in order], "cardinalities": cardinalities,
            "regions": regions, "confidence": confidence,
            "discarded": discarded,
            "irrelevant_unknown": [j for j, w in weights.items() if w == 0 and j not in oracle.facts],
            "initial_mask_index": initial_index, "final_mask_index": index,
            "initial_retained_unknown_count": len(initial_retained),
            "initial_retained_unknown": sorted(initial_retained),
            "request_containment_checked": True,
            "containment_scope": "post-cut, post-audit residual execution identities",
            "conflict_freeze": conflict, "conflict_reason": conflict_reason,
            "final_scaled_support": final_support, "projection_calls": envelope.calls,
            "support_decisions": envelope.decisions, "events": events, "output": output,
            "timings": {"preprocessing_seconds": preprocessing_seconds,
                        "graph_cut_seconds": graph_cut_seconds,
                        "affine_decomposition_seconds": affine_seconds,
                        "audit_seconds": audit_seconds,
                        "audit_inversion_seconds": inversion_seconds,
                        "projection_seconds": projection_seconds,
                        "execution_bookkeeping_seconds": bookkeeping_seconds,
                        "oracle_acquisition_bookkeeping_seconds": acquisition_seconds,
                        "algorithm_seconds": algorithm_seconds,
                        "sql_evaluation_seconds": materialization_seconds}}
