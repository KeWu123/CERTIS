"""Exact snapshot mask allocation for key-disjoint unit-incidence joins.

The min-plus recurrence is classical. Exchangeable within-key counts reduce
the all-identity-mask problem to that recurrence. Unit request prices and
uncoupled side/key intervals are essential. Repeated identities are rejected
by the caller, which owns incidence validation and identity reconstruction.

Uses exact Python integers as a reference/fallback. A checked int64 path
vectorizes the same arithmetic; no floating-point certificate decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from bilinear_certificate import InconsistentEvidence, RegionSide


@dataclass(frozen=True)
class SideCounts:
    lower: int
    upper: int
    positive: int
    unknown: int
    forced_discard: int = 0

    def __post_init__(self):
        for field in ("lower", "upper", "positive", "unknown", "forced_discard"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field} must be a Python integer")
            if value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if self.forced_discard > self.unknown:
            raise ValueError("Forced discards exceed unknown population")
        if self.lower > self.upper or self.upper < self.positive or self.lower > self.positive + self.unknown:
            raise InconsistentEvidence("Empty count evidence")


@dataclass(frozen=True)
class CountProfile:
    forced: int
    values: tuple[int, ...]
    choices: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CountPlan:
    discarded: int
    support: int
    counts: tuple[tuple[int, int], ...]
    minimum_support_by_extra_count: tuple[int, ...]
    forced_discarded: int
    backend: str


def _fraction(value):
    if not isinstance(value, Fraction):
        raise TypeError("Allowance must be fractions.Fraction")
    if not 0 <= value <= 1:
        raise ValueError("Allowance must be in [0,1]")
    return value.numerator, value.denominator


def frontier(side: RegionSide) -> tuple[tuple[int, int], ...]:
    """At most two exact (lost, retained) maximization candidates.

For fixed total t, the smallest retained count is max(h,t-A). Below
A+h this is constant and increasing total cannot reduce the objective.
Above A+h the objective is affine in this side's total. Only the clipped
breakpoint and greatest feasible total are needed. Nonnegative q-p and
opposite retained count, and the loss allowance in [0,1], are essential.
"""
    if not isinstance(side, RegionSide):
        raise TypeError("RegionSide required")
    A, C, h = side.unknown_lost, side.unknown_kept, side.known_positive
    low, high = max(side.lower_total, h), min(side.upper_total, h + A + C)
    if low > high:
        raise InconsistentEvidence("Empty frontier")
    start = min(high, max(low, A + h))
    return tuple(dict.fromkeys((t - max(h, t - A), max(h, t - A)) for t in (start, high)))


def frontier_support(left: RegionSide, right: RegionSide, allowance: Fraction) -> int:
    p, q = _fraction(allowance)
    return max((q-p)*(a+b)*(c+d)-q*b*d
               for a, b in frontier(left) for c, d in frontier(right))


def _python_profile(left, right, allowance):
    ml, mr = left.forced_discard, right.forced_discard
    size = left.unknown + right.unknown - ml - mr + 1
    values, choices = [None]*size, [None]*size
    for dl in range(ml, left.unknown + 1):
        vl = RegionSide(left.lower, left.upper, left.positive, dl, left.unknown-dl)
        for dr in range(mr, right.unknown + 1):
            vr = RegionSide(right.lower, right.upper, right.positive, dr, right.unknown-dr)
            value = frontier_support(vl, vr, allowance)
            extra = dl+dr-ml-mr
            if values[extra] is None or value < values[extra]:
                values[extra], choices[extra] = value, (dl, dr)
    return CountProfile(ml+mr, tuple(values), tuple(choices))


def _int64_safe(keys, p, q):
    # Includes signed sums and intermediate products with a generous margin;
    # the numerical denominator never indexes a DP array.
    bound = q * sum((a.positive+a.unknown)*(b.positive+b.unknown) for a, b in keys)
    scalar_bound = q * max((s.positive+s.unknown for pair in keys for s in pair), default=1)
    return max(p, q, bound, scalar_bound, 0) <= np.iinfo(np.int64).max//8


def _numpy_profile(left, right, p, q):
    ml, mr = left.forced_discard, right.forced_discard
    dr = np.arange(mr, right.unknown+1, dtype=np.int64)
    rlow = max(right.lower, right.positive)
    rhigh = min(right.upper, right.positive+right.unknown)
    rt0 = np.minimum(rhigh, np.maximum(rlow, dr+right.positive))
    rt1 = np.full_like(dr, rhigh)
    rb0 = np.maximum(right.positive, rt0-dr)
    rb1 = np.maximum(right.positive, rt1-dr)
    size = left.unknown+right.unknown-ml-mr+1
    values = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    choices_l, choices_r = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.int64)
    llow, lhigh = max(left.lower, left.positive), min(left.upper, left.positive+left.unknown)
    for dl in range(ml, left.unknown+1):
        lt0 = min(lhigh, max(llow, dl+left.positive))
        lt1 = lhigh
        lb0, lb1 = max(left.positive, lt0-dl), max(left.positive, lt1-dl)
        value = (q-p)*lt0*rt0-q*lb0*rb0
        for lt, lb, rt, rb in ((lt0,lb0,rt1,rb1), (lt1,lb1,rt0,rb0), (lt1,lb1,rt1,rb1)):
            value = np.maximum(value, (q-p)*lt*rt-q*lb*rb)
        positions = dl+dr-ml-mr
        better = value < values[positions]
        selected = positions[better]
        values[selected] = value[better]
        choices_l[selected], choices_r[selected] = dl, dr[better]
    if np.any(values == np.iinfo(np.int64).max):
        raise AssertionError("Incomplete local count profile")
    return CountProfile(ml+mr, tuple(map(int,values)),
                        tuple(zip(map(int,choices_l), map(int,choices_r))))


def plan_counts(keys, allowance: Fraction, *, backend="auto") -> CountPlan | None:
    """Maximize certified unknown discards, containing forced local sets.

    Returns None when the imposed old discards have no certifying superset.
    Nonempty evidence is checked by SideCounts, including orphan sides.
    Identity sets must be reconstructed to contain the actual old sets;
    count lower bounds alone do not imply set containment.
    """
    p, q = _fraction(allowance)
    keys = tuple(tuple(pair) for pair in keys)
    if any(len(pair) != 2 or any(not isinstance(s, SideCounts) for s in pair) for pair in keys):
        raise TypeError("Pairs of SideCounts required")
    if backend not in ("auto", "python", "numpy"):
        raise ValueError("Unknown arithmetic backend")
    safe = _int64_safe(keys, p, q)
    if backend == "numpy" and not safe:
        raise OverflowError("int64 arithmetic bound exceeded; use Python integers")
    use_numpy = backend != "python" and safe
    profiles = tuple(_numpy_profile(a,b,p,q) if use_numpy else _python_profile(a,b,allowance)
                     for a,b in keys)
    offset = sum(pr.values[0] for pr in profiles if len(pr.values)==1)
    active = [(i,pr) for i,pr in enumerate(profiles) if len(pr.values)>1]
    scores = np.array([offset],dtype=np.int64) if use_numpy else [offset]
    back = []
    for _, pr in active:
        size = len(scores)+len(pr.values)-1
        if use_numpy:
            new = np.full(size,np.iinfo(np.int64).max,dtype=np.int64)
            take = np.zeros(size,dtype=np.int64)
            for extra,value in enumerate(pr.values):
                candidate = scores+value
                segment = new[extra:extra+len(scores)]
                better = candidate < segment
                segment[better] = candidate[better]
                take_segment = take[extra:extra+len(scores)]
                take_segment[better] = extra
        else:
            new, take = [None]*size, [None]*size
            for extra,value in enumerate(pr.values):
                for used,previous in enumerate(scores):
                    target, candidate = extra+used, value+previous
                    if new[target] is None or candidate < new[target]:
                        new[target], take[target] = candidate, extra
        scores = new
        back.append(take)
    feasible = [i for i,value in enumerate(scores) if value <= 0]
    if not feasible:
        return None
    chosen, counts = feasible[-1], [pr.choices[0] for pr in profiles]
    cursor = chosen
    for (index,pr),take in zip(reversed(active),reversed(back)):
        extra = int(take[cursor])
        counts[index] = pr.choices[extra]
        cursor -= extra
    if cursor:
        raise AssertionError("Invalid allocation reconstruction")
    forced = sum(pr.forced for pr in profiles)
    return CountPlan(forced+chosen,int(scores[chosen]),tuple(counts),
                     tuple(map(int,scores)),forced,"numpy-int64" if use_numpy else "python-integer")
