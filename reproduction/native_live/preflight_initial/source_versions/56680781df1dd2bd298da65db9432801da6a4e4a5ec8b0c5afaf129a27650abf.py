"""Exact local count profiles from piecewise quadratic envelopes.

This is an opt-in builder. It returns the reference CountProfile, including
the lexicographically first attaining (left, right) counts. All branch,
root-neighbour and certificate comparisons use arbitrary-precision integers.
The existing certificate and allocation kernels are unchanged.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isqrt

from bilinear_certificate import RegionSide
from count_mask_planner import CountProfile, SideCounts, _fraction, frontier_support


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _product(a, b):
    return a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1]


def _root_floors(poly):
    """Floors of all real roots of C+B*x+A*x*x, exactly.

    For a positive integer denominator, replacing sqrt(D) by its floor
    (positive root) or ceiling (negative root) preserves the quotient floor.
    Constant and identically zero polynomials create no isolated boundary.
    """
    c, b, a = poly
    if a == 0:
        if b == 0:
            return ()
        return ((-c)//b,)
    if a < 0:
        a, b, c = -a, -b, -c
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return ()
    s = isqrt(discriminant)
    ceiling = s+(s*s != discriminant)
    return tuple(dict.fromkeys(((-b-ceiling)//(2*a), (-b+s)//(2*a))))


def _affine_frontier(side, discard, midpoint2):
    """Return affine lost, low-retained, high-retained counts on a piece."""
    intercept, slope = discard
    low = max(side.lower, side.positive)
    high = min(side.upper, side.positive+side.unknown)
    count2 = 2*intercept+slope*midpoint2
    lost = discard if count2 <= 2*(high-side.positive) else (high-side.positive, 0)
    retained = tuple((total-intercept, -slope) if count2 <= 2*(total-side.positive)
                     else (side.positive, 0) for total in (low, high))
    return lost, retained


def _polynomials(left, right, g, lo, hi, p, q):
    a, bs = _affine_frontier(left, (0, 1), lo+hi)
    c, ds = _affine_frontier(right, (g, -1), lo+hi)
    result = []
    for b in bs:
        for d in ds:
            total = _product(_add(a, b), _add(c, d))
            retained = _product(b, d)
            polynomial = tuple((q-p)*v-q*w for v, w in zip(total, retained))
            if polynomial not in result:
                result.append(polynomial)
    return result


def _piece_minimum(polynomials, lo, hi, metrics=None):
    """Minimize an upper envelope over integer x, breaking ties toward lo."""
    candidates = {lo, hi}

    def include_floor(value):
        for x in (value, value+1):
            if lo <= x <= hi:
                candidates.add(x)

    for c, b, a in polynomials:
        if a:
            include_floor((-b)//(2*a))
    for first, second in combinations(polynomials, 2):
        difference = tuple(v-w for v, w in zip(first, second))
        for floor in _root_floors(difference):
            include_floor(floor)
    best = min((max((a*x+b)*x+c for c, b, a in polynomials), x) for x in candidates)
    if metrics is not None:
        metrics['pieces'] += 1
        metrics['polynomials'] += len(polynomials)
        metrics['candidate_evaluations'] += len(candidates)
    return best


def piecewise_profile(left: SideCounts, right: SideCounts, allowance: Fraction,
                      *, metrics=None, verify_witness=True) -> CountProfile:
    """Build an identical exact profile with O(n_L+n_R+1) candidate searches.

    The arithmetic-operation bound is linear; integer multiplication and
    isqrt retain their operand-dependent bit costs. Forced old discards are
    count lower bounds. The caller remains responsible for identity-level
    containment when reconstructing the mask.
    """
    if not isinstance(left, SideCounts) or not isinstance(right, SideCounts):
        raise TypeError('Two SideCounts required')
    p, q = _fraction(allowance)
    if metrics is not None:
        for name in ('profile_entries', 'pieces', 'polynomials', 'candidate_evaluations'):
            metrics.setdefault(name, 0)
    left_breaks = (max(left.lower, left.positive)-left.positive,
                   min(left.upper, left.positive+left.unknown)-left.positive)
    right_breaks = (max(right.lower, right.positive)-right.positive,
                    min(right.upper, right.positive+right.unknown)-right.positive)
    forced = left.forced_discard+right.forced_discard
    values, choices = [], []
    for g in range(forced, left.unknown+right.unknown+1):
        lower = max(left.forced_discard, g-right.unknown)
        upper = min(left.unknown, g-right.forced_discard)
        breaks = sorted({lower, upper}|{x for x in (*left_breaks, *(g-v for v in right_breaks))
                                        if lower < x < upper})
        if lower == upper:
            dl, dr = lower, g-lower
            value = frontier_support(RegionSide(left.lower,left.upper,left.positive,dl,left.unknown-dl),
                                     RegionSide(right.lower,right.upper,right.positive,dr,right.unknown-dr),allowance)
            best = value, lower
        else:
            best = min(_piece_minimum(_polynomials(left,right,g,lo,hi,p,q),lo,hi,metrics)
                       for lo, hi in zip(breaks, breaks[1:]))
        value, dl = best
        dr = g-dl
        if verify_witness:
            actual = frontier_support(RegionSide(left.lower,left.upper,left.positive,dl,left.unknown-dl),
                                      RegionSide(right.lower,right.upper,right.positive,dr,right.unknown-dr),allowance)
            if value != actual:
                raise AssertionError('Polynomial envelope disagrees with the four-pair kernel')
        values.append(value)
        choices.append((dl, dr))
    if metrics is not None:
        metrics['profile_entries'] += len(values)
    return CountProfile(forced, tuple(values), tuple(choices))
