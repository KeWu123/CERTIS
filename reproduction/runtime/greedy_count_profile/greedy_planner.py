"""Deterministic marginal-greedy allocation over unmodified exact profiles.

This is a certificate-safe heuristic. It does not optimize total cardinality.
Only the global min-plus convolution of the checked planner is replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code'))
import count_mask_planner as exact

PROFILE_SHA256 = '42b68da0756eba1ee6bb41a00139e25dcd27fbc0a66e42096ee93edafe25d01d'


@dataclass(frozen=True)
class GreedyPlan:
    discarded: int
    support: int
    counts: tuple[tuple[int, int], ...]
    forced_discarded: int
    backend: str
    chosen_extra_counts: tuple[int, ...]


def verify_profile_source():
    path = Path(exact.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROFILE_SHA256:
        raise AssertionError('Exact local profile source does not match the pinned kernel')


def allocate_profiles(profiles, *, backend='python-integer'):
    """Choose counts; heap ties are broken by the original key index.

    Every support value and marginal is a Python integer, even when the
    guarded local profile was computed with int64. Signed starting supports
    include all known-only negative credits. A noncertifiable start returns
    None. Local profile monotonicity means no larger count can repair it.
    """
    profiles = tuple(profiles)
    selected = [0] * len(profiles)
    support = sum(pr.values[0] for pr in profiles)
    if support > 0:
        return None
    queue = []
    for index, pr in enumerate(profiles):
        if len(pr.values) > 1:
            heapq.heappush(queue, (pr.values[1] - pr.values[0], index))
    while queue:
        marginal, index = heapq.heappop(queue)
        if marginal < 0:
            raise AssertionError('Exact local count support must be nondecreasing')
        if support + marginal > 0:
            break  # All currently accessible increments have at least this cost.
        support += marginal
        selected[index] += 1
        count = selected[index]
        pr = profiles[index]
        if count + 1 < len(pr.values):
            heapq.heappush(queue, (pr.values[count+1] - pr.values[count], index))
    forced = sum(pr.forced for pr in profiles)
    return GreedyPlan(forced + sum(selected), support,
                      tuple(pr.choices[k] for pr, k in zip(profiles, selected)),
                      forced, backend, tuple(selected))


def plan_counts(keys, allowance: Fraction, *, backend='auto') -> GreedyPlan | None:
    """Use the exact planner's validation, overflow guard and profile kernels."""
    verify_profile_source()
    p, q = exact._fraction(allowance)
    keys = tuple(tuple(pair) for pair in keys)
    if any(len(pair) != 2 or any(not isinstance(s, exact.SideCounts) for s in pair) for pair in keys):
        raise TypeError('Pairs of SideCounts required')
    if backend not in ('auto', 'python', 'numpy'):
        raise ValueError('Unknown arithmetic backend')
    safe = exact._int64_safe(keys, p, q)
    if backend == 'numpy' and not safe:
        raise OverflowError('int64 arithmetic bound exceeded; use Python integers')
    use_numpy = backend != 'python' and safe
    profiles = tuple(exact._numpy_profile(a, b, p, q) if use_numpy
                     else exact._python_profile(a, b, allowance) for a, b in keys)
    return allocate_profiles(profiles, backend='numpy-int64' if use_numpy else 'python-integer')
