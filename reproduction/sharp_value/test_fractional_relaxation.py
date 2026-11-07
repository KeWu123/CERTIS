"""Independently enumerate small multiple-choice LP vertices with Fraction.

For K keys, there are K independent normalization equalities, one per key.
If the single budget inequality is active, a basic feasible vertex has at
most K+1 positive option weights; every key needs at least one.  Thus a vertex
uses pure options for all keys except, at most, one key mixing two options.
With slack budget all keys can be pure.  Enumerating these cases gives the
exact LP optimum without using a convex hull or marginal-cost ordering.

These are synthetic arithmetic tests, not query workloads or live timings.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
import random

from fractional_bound import fractional_upper_bound


@dataclass(frozen=True)
class Profile:
    forced: int
    values: tuple[int, ...]


def vertex_optimum(profiles):
    """Maximize forced + extra discards over every small exact LP vertex."""
    profiles = tuple(profiles)
    forced = sum(profile.forced for profile in profiles)
    best = None
    # Vertices with no mixed key, including optima with slack budget.
    for choices in product(*(range(len(profile.values)) for profile in profiles)):
        if sum(profile.values[g] for profile, g in zip(profiles, choices)) <= 0:
            value = Fraction(forced + sum(choices))
            best = value if best is None else max(best, value)

    # All remaining vertices: one key mixes an arbitrary pair of options,
    # not merely adjacent options or points on the production lower hull.
    for mixed_key, profile in enumerate(profiles):
        others = [index for index in range(len(profiles)) if index != mixed_key]
        for choices in product(*(range(len(profiles[index].values)) for index in others)):
            available = -sum(profiles[index].values[g] for index, g in zip(others, choices))
            for lo in range(len(profile.values)):
                for hi in range(lo + 1, len(profile.values)):
                    lower, upper = profile.values[lo], profile.values[hi]
                    if lower < upper and lower <= available <= upper:
                        value = (forced + sum(choices) + lo
                                 + Fraction((available - lower) * (hi - lo), upper - lower))
                        best = value if best is None else max(best, value)
                    # Equal-cost pairs need no fractional candidate: their
                    # larger-cardinality pure endpoint was enumerated above.
    return best


def check(profiles):
    expected = vertex_optimum(profiles)
    observed = fractional_upper_bound(profiles)
    assert observed == expected, (profiles, observed, expected)
    if observed is not None:
        assert isinstance(observed, Fraction)
    return observed


def run():
    explicit = [
        (),
        (Profile(0, (0,)),),
        (Profile(2, (0, 0, 0)),),             # Zero budget and zero-cost edges.
        (Profile(0, (0, 3, 3)),),             # A plateau after an unaffordable entry.
        (Profile(4, (-1, 2)),),               # Noninteger optimum 4 + 1/3.
        (Profile(2, (-4, -4, -4, 8)),),       # Free prefix, then a fractional edge.
        (Profile(3, (-2, 0, 6)), Profile(1, (1, 2, 2))),
        (Profile(1, (-5, -5, 0)), Profile(2, (5, 5, 8))),
        (Profile(0, (1, 2)), Profile(3, (1, 1))),  # Infeasible minimum support.
        (Profile(1, (-8, -7, -6)), Profile(2, (1, 2))),  # Slack at all maxima.
    ]
    for profiles in explicit:
        check(profiles)
    assert check((Profile(4, (-1, 2)),)) == Fraction(13, 3)

    # Preserve the seed and generation used in the independent small review.
    rng = random.Random(7391)
    for _ in range(160):
        profiles = []
        for _ in range(rng.randint(1, 3)):
            values = [rng.randint(-8, 6)]
            for _ in range(rng.randint(0, 3)):
                values.append(values[-1] + rng.randint(0, 8))
            profiles.append(Profile(rng.randint(0, 3), tuple(values)))
        check(profiles)
    return dict(
        status="pass", exact_lp_vertex_cases=160 + len(explicit),
        deterministic_random_cases=160, explicit_edge_cases=len(explicit), seed=7391,
        maximum_random_keys=3, maximum_random_options_per_key=4,
        check="Exact Fraction equality with independent pure/one-mixed-key LP vertex enumeration",
        scope="Monotone incident-count profiles including forced discards; no model, GPU or query timing.",
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
