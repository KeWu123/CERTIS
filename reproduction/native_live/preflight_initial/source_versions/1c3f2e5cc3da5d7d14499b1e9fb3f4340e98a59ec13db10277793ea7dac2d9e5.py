"""Pure-integer bag-recall support for complete equality-join blocks.

Each side has a truthful cache, all known positives are rescued, and unknown
identities are split by one effective discard mask. Counts are sufficient
only for complete unit-multiplicity bipartite blocks. ``single_key_exact``
requires identities to belong to one key and no retained cross-key count
constraints. Repeated identities give a safe relaxation by forgetting their
cross-key equalities; every known fact must be propagated to every region.

This module does not construct confidence intervals or validate identity
membership. The caller supplies simultaneously covered region intervals and
declares the identity mode explicitly. Python integers avoid overflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Literal

IdentityMode = Literal["single_key_exact", "repeated_identity_relaxation"]


class InconsistentEvidence(ValueError):
    """The supplied count interval and truthful point facts have no completion."""


def _nonnegative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer (no float or bool coercion)")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RegionSide:
    lower_total: int
    upper_total: int
    known_positive: int
    unknown_lost: int
    unknown_kept: int

    def __post_init__(self) -> None:
        for name in ("lower_total", "upper_total", "known_positive",
                     "unknown_lost", "unknown_kept"):
            _nonnegative_integer(name, getattr(self, name))


def polygon_vertices(side: RegionSide) -> tuple[tuple[int, int], ...]:
    """Return exact vertices (lost positives, retained positives), at most six.

    Point and segment domains are included. Raises InconsistentEvidence for
    an empty domain; empty evidence must never become vacuous certification.
    """
    if not isinstance(side, RegionSide):
        raise TypeError("side must be a RegionSide")
    A, K, h = side.unknown_lost, side.unknown_kept, side.known_positive
    lo, hi = side.lower_total, side.upper_total
    if lo > hi or hi < h or lo > h+A+K:
        raise InconsistentEvidence("Count interval conflicts with region capacities/cache")
    lines = ((1, 0, 0), (1, 0, A), (0, 1, h), (0, 1, h+K),
             (1, 1, lo), (1, 1, hi))
    points: set[tuple[int, int]] = set()
    for (x, y, c), (u, v, d) in combinations(lines, 2):
        det = x*v-u*y
        if det == 0:
            continue
        # The only nonzero determinants are +/-1, hence exact integer vertices.
        a, b = (c*v-d*y)//det, (x*d-u*c)//det
        if 0 <= a <= A and h <= b <= h+K and lo <= a+b <= hi:
            points.add((a, b))
    if not points:
        raise InconsistentEvidence("No feasible count completion")
    return tuple(sorted(points))


def _allowance(allowance: Fraction) -> tuple[int, int]:
    if not isinstance(allowance, Fraction):
        raise TypeError("allowance must be fractions.Fraction")
    if allowance < 0 or allowance > 1:
        raise ValueError("allowance must be in [0, 1]")
    return allowance.numerator, allowance.denominator


def key_support(left: RegionSide, right: RegionSide, allowance: Fraction) -> int:
    """max [(q-p)*total-q*retained] for p/q=allowance, using <=36 pairs."""
    p, q = _allowance(allowance)
    vl, vr = polygon_vertices(left), polygon_vertices(right)
    return max((q-p)*(a+b)*(c+d)-q*b*d for a, b in vl for c, d in vr)


@dataclass(frozen=True)
class SupportResult:
    scaled_support: int
    allowance: Fraction
    mode: IdentityMode
    key_count: int

    @property
    def certified(self) -> bool:
        return self.scaled_support <= 0

    @property
    def is_sharp(self) -> bool:
        """Sharp relative to supplied independent intervals, in declared mode."""
        return self.mode == "single_key_exact"


def support_sum(
    keys: Iterable[tuple[RegionSide, RegionSide]],
    allowance: Fraction,
    *,
    mode: IdentityMode,
) -> SupportResult:
    """Sum signed supports, with explicit exact-vs-relaxation metadata.

    Do not clamp negative key supports: a safely retained key can offset loss
    elsewhere. The empty key collection has support zero and certifies the
    empty query. ``mode`` is a caller assertion; counts cannot detect reuse.
    """
    _allowance(allowance)
    if mode not in ("single_key_exact", "repeated_identity_relaxation"):
        raise ValueError("Unknown identity mode")
    total, count = 0, 0
    for left, right in keys:
        total += key_support(left, right, allowance)
        count += 1
    return SupportResult(total, allowance, mode, count)
