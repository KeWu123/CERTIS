"""Exact per-side recall frontiers and forced-discard-aware split updates."""
from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass


@dataclass(frozen=True)
class Side:
    lower: int
    upper: int
    positive: int
    unknown: int
    forced: int = 0

    def bounds(self):
        low = max(self.lower, self.positive)
        high = min(self.upper, self.positive + self.unknown)
        if min(self.lower, self.upper, self.positive, self.unknown, self.forced) < 0:
            raise ValueError('Nonnegative integer counts required')
        if low > high or self.forced > self.unknown:
            raise ValueError('Inconsistent side evidence')
        return low, high


def recall_pair(side: Side, discard: int) -> tuple[int, int]:
    """Minimum retained-positive fraction; empty total has recall one."""
    low, high = side.bounds()
    if not side.forced <= discard <= side.unknown:
        raise ValueError('Discard outside forced/capacity range')
    if high == 0 or discard == 0:
        return 1, 1
    total = min(high, max(low, 1, side.positive + discard))
    retained = total - min(discard, total - side.positive)
    return retained, total


def satisfies(a, b, target=Fraction(9, 10)):
    return a[0] * b[0] * target.denominator >= a[1] * b[1] * target.numerator


def continuous_plan(left: Side, right: Side, target=Fraction(9, 10)):
    """Best integer pair over all continuous splits, tie: smaller left count.

    Each per-side recall frontier is nonincreasing in discarded count, so a
    two-pointer scan visits at most n_left+n_right+2 pairs. No model labels
    outside the input count evidence are read.
    """
    left.bounds(); right.bounds()
    rd = right.unknown
    best = None
    for ld in range(left.forced, left.unknown + 1):
        lr = recall_pair(left, ld)
        while rd >= right.forced and not satisfies(lr, recall_pair(right, rd), target):
            rd -= 1
        if rd < right.forced:
            break
        if best is None or ld + rd > sum(best):
            best = ld, rd
    return best


def cap_at_recall(side: Side, required: Fraction):
    lo, hi = 0, side.unknown
    unforced = Side(side.lower, side.upper, side.positive, side.unknown)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        a, b = recall_pair(unforced, mid)
        if a * required.denominator >= b * required.numerator:
            lo = mid
        else:
            hi = mid - 1
    return lo


def grid_plan(left: Side, right: Side, target=Fraction(9, 10), fixed=None):
    """Original 0.005 split grid; all accepted counts are rechecked exactly."""
    choices = [fixed] if fixed is not None else [
        (Fraction(i, 200), 1-target/(1-Fraction(i, 200)))
        for i in range(1, 20)
        if 1-target/(1-Fraction(i, 200)) >= Fraction(49, 10000)]
    best = split = None
    for el, er in choices:
        ld = cap_at_recall(left, 1-el)
        rd = cap_at_recall(right, 1-er)
        if ld < left.forced or rd < right.forced:
            continue
        if not satisfies(recall_pair(left, ld), recall_pair(right, rd), target):
            raise AssertionError('Grid accepted unsafe product recall')
        if best is None or ld+rd > sum(best):
            best, split = (ld, rd), (el, er)
    return best, split
