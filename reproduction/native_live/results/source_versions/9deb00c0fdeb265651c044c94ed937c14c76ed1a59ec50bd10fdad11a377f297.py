"""Integer comparisons for equal-tail hypergeometric inversion.

alpha is the exact supplied IEEE float value, interpreted by as_integer_ratio.
No SciPy floating tail value decides inclusion of an endpoint.
"""
from functools import lru_cache
from fractions import Fraction
import math
import numbers
from scipy.stats import hypergeom


@lru_cache(maxsize=1024)
def _denominator(population, sample):
    return math.comb(population, sample)


def tail_at_least(population, successes, sample, observed, upper, cutoff):
    """Compare an exact hypergeometric tail with a rational cutoff."""
    low, high = max(0, sample - population + successes), min(sample, successes)
    if upper:
        if observed <= low:
            return True
        if observed > high:
            return False
        start, stop, step = high, observed, -1
    else:
        if observed >= high:
            return True
        if observed < low:
            return False
        start, stop, step = low, observed, 1
    p, q = cutoff.numerator, cutoff.denominator
    target = p * _denominator(population, sample)
    term = math.comb(successes, start) * math.comb(population - successes, sample - start)
    total, k = 0, start
    while True:
        total += term
        if total * q >= target:
            return True
        if k == stop:
            return False
        if step == 1:
            term = term * (successes - k) * (sample - k) // ((k + 1) * (population - successes - sample + k + 1))
        else:
            term = term * k * (population - successes - sample + k) // ((successes - k + 1) * (sample - k + 1))
        k += step


@lru_cache(maxsize=65536)
def count_interval_exact(population, sample, positives, alpha):
    if not all(isinstance(v, numbers.Integral) for v in (population, sample, positives)):
        raise ValueError("Integer population, sample and positive counts required")
    population, sample, positives = map(int, (population, sample, positives))
    if not population >= 1 or not 0 <= positives <= sample <= population:
        raise ValueError("Invalid finite-population counts")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Invalid failure probability")
    if sample == 0:
        return 0, population
    if sample == population:
        return positives, positives
    cutoff = Fraction.from_float(alpha) / 2
    possible_lo, possible_hi = positives, population - sample + positives
    left, right = possible_lo, possible_hi
    while left < right:
        mid = (left + right) // 2
        if hypergeom.sf(positives - 1, population, mid, sample) >= alpha / 2:
            right = mid
        else:
            left = mid + 1
    lower = left
    # Floating tails propose endpoints; EXACT comparisons alone accept them.
    # Monotonicity makes adjustment in either direction complete, even if a
    # floating proposal is far from the true boundary.
    while lower > possible_lo and tail_at_least(population, lower - 1, sample, positives, True, cutoff):
        lower -= 1
    while lower <= possible_hi and not tail_at_least(population, lower, sample, positives, True, cutoff):
        lower += 1
    left, right = possible_lo, possible_hi
    while left < right:
        mid = (left + right + 1) // 2
        if hypergeom.cdf(positives, population, mid, sample) >= alpha / 2:
            left = mid
        else:
            right = mid - 1
    upper = left
    while upper < possible_hi and tail_at_least(population, upper + 1, sample, positives, False, cutoff):
        upper += 1
    while upper >= possible_lo and not tail_at_least(population, upper, sample, positives, False, cutoff):
        upper -= 1
    if lower > upper:
        raise ArithmeticError("Raw inversion unexpectedly empty")
    return lower, upper


def allocated_alpha(delta, events):
    """Allocate a budget whose exact value never exceeds delta/m.

    Rational inputs are preserved exactly: the budget is delta divided by
    the integer event count, then rounded DOWN to the nearest representable
    float. A float input's exact binary value defines the nominal budget, so
    `allocated_alpha(0.1, 2)` returns the largest float <= 1/20. In either
    case the returned float satisfies `Fraction.from_float(alpha) <= delta/events`.
    """
    if not isinstance(events, numbers.Integral) or events < 1:
        raise ValueError("Positive integer event count required")
    if not isinstance(delta, numbers.Rational) and not isinstance(delta, float):
        raise ValueError("delta must be a real number")
    if isinstance(delta, numbers.Integral):
        delta = float(delta)
    if not math.isfinite(float(delta)) or not 0 < float(delta) < 1:
        raise ValueError("Invalid workload budget")
    target = Fraction(delta) / int(events) if isinstance(delta, numbers.Rational) \
        else Fraction.from_float(delta) / int(events)
    alpha = float(target)
    if Fraction.from_float(alpha) > target:
        alpha = math.nextafter(alpha, 0.)
    if alpha == 0:
        raise ValueError("Error allocation underflows; use a rational budget interface")
    return alpha
