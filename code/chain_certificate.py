"""Chain composition of bag-recall certificates -- research candidate.

Independent of the manuscript and of the earlier support primitive.  The
reference path enumerates Boolean label vectors and discard masks directly
and recomputes the composed bag from actual tuple multiplicities, so it does
not reuse the polygon or frontier code.

Model.  A *chain* joins F relations S_0,...,S_{F-1} on one common key with
complete unit incidence: the answer bag contains every tuple
(i_0,...,i_{F-1}) whose F unary labels are all positive.  With
X_t = sum_{i in S_t} z_i the full mass is T = prod_t X_t, and for a discard
mask G the ideal retained mass is B = prod_t X'_t, where X'_t counts retained
positives of relation t (cached positives are rescued).  The signed objective
at allowance p/q is (q-p)T - qB, and support <= 0 certifies.  Equivalently the
certified allowance of a mask is min over compatible worlds of B/T, taken
over worlds with T > 0.

Relation t carries: count interval [lo,hi], known positives h, unknown
population u, discard count d <= u, and `old` forced discards.  Identities are
key disjoint, so all statements are sharp relative to these intervals.
"""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction
from itertools import product
from time import perf_counter

sys.dont_write_bytecode = True


class InconsistentEvidence(ValueError):
    """The interval admits no side completion."""


class Side:
    """One relation of the chain at the current cache and mask."""

    __slots__ = ("lo", "hi", "h", "u", "d")

    def __init__(self, lo: int, hi: int, h: int, u: int, d: int = 0) -> None:
        for name, v in (("lo", lo), ("hi", hi), ("h", h), ("u", u), ("d", d)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if lo > hi or hi < h or lo > h + u or hi > h + u:
            raise InconsistentEvidence("side interval conflicts with capacity")
        if d > u:
            raise ValueError("discard count exceeds unknown population")
        self.lo, self.hi, self.h, self.u, self.d = lo, hi, h, u, d

    @property
    def A(self) -> int:
        return self.d

    @property
    def K(self) -> int:
        return self.u - self.d

    # -- exact geometry -------------------------------------------------
    def vertices(self) -> tuple[tuple[int, int], ...]:
        """All integral vertices (lost positives a, retained positives b)."""
        A, K, h, lo, hi = self.A, self.K, self.h, self.lo, self.hi
        lines = ((1, 0, 0), (1, 0, A), (0, 1, h), (0, 1, h + K),
                 (1, 1, lo), (1, 1, hi))
        pts: set[tuple[int, int]] = set()
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x, y, c = lines[i]
                x2, y2, c2 = lines[j]
                det = x * y2 - x2 * y
                if det == 0:
                    continue
                a, b = (c * y2 - c2 * y) // det, (x * c2 - x2 * c) // det
                if 0 <= a <= A and h <= b <= h + K and lo <= a + b <= hi:
                    pts.add((a, b))
        if not pts:
            raise InconsistentEvidence("no feasible side completion")
        return tuple(sorted(pts))

    def frontier(self) -> tuple[tuple[int, int], ...]:
        """Candidate points (a,b) = (t-b(t), b(t)) at t in {t*, t_max}."""
        A, K, h = self.A, self.K, self.h
        t_min = max(self.lo, h)
        t_max = min(self.hi, h + A + K)
        t_star = min(t_max, max(t_min, A + h))
        out = set()
        for t in (t_star, t_max):
            b = max(h, t - A)
            out.add((t - b, b))
        return tuple(sorted(out))

def _objective(counts, p: int, q: int) -> int:
    total = kept = 1
    for a, b in counts:
        total *= a + b
        kept *= b
    return (q - p) * total - q * kept


def support_exact(sides, allowance: Fraction) -> int:
    """Maximum over the full Cartesian product of per-side vertex sets."""
    p, q = allowance.numerator, allowance.denominator
    return max(_objective(c, p, q) for c in product(*[s.vertices() for s in sides]))


def support_frontier(sides, allowance: Fraction) -> int:
    """Maximum over the <=2^F endpoint frontier."""
    p, q = allowance.numerator, allowance.denominator
    return max(_objective(c, p, q) for c in product(*[s.frontier() for s in sides]))


def _labellings(side: Side):
    """All label vectors compatible with the interval and the forced mask."""
    for bits in product((0, 1), repeat=side.u):
        z = (1,) * side.h + bits
        total = sum(z)
        if not (side.lo <= total <= side.hi):
            continue
        kept = sum(zz * (1 - gg)
                   for zz, gg in zip(z, [0] * side.h
                                     + [1 if i < side.d else 0
                                        for i in range(side.u)]))
        yield total, kept, z, bits


def support_bruteforce(sides, p: int, q: int):
    """Enumerate every labelling and count actual composed tuples."""
    per = [list(_labellings(s)) for s in sides]
    if any(not v for v in per):
        raise InconsistentEvidence("no compatible labelling for a side")
    best, arg = None, None
    for combo in product(*per):
        total = kept = 1
        for t, k, _z, _b in combo:
            total *= t
            kept *= k
        value = (q - p) * total - q * kept
        if best is None or value > best:
            best, arg = value, [c[3] for c in combo]
    return best, arg


def worst_case_loss(sides) -> Fraction:
    """max over compatible worlds of (T-B)/T, the mask's worst-case loss.

    A mask is certified at allowance lambda exactly when this value <= lambda.
    Worlds with T = 0 contribute loss 0 by the zero-answer convention.
    """
    per = [list(_labellings(s)) for s in sides]
    if any(not v for v in per):
        raise InconsistentEvidence("no compatible labelling for a side")
    worst = Fraction(0)
    for combo in product(*per):
        total = kept = 1
        for t, k, _z, _b in combo:
            total *= t
            kept *= k
        if total == 0:
            continue
        loss = 1 - Fraction(kept, total)
        if loss > worst:
            worst = loss
    return worst


def staged_loss(sides) -> Fraction:
    """Union-bound composition of the per-block worst-case losses.

    For F relations the block t = (t, t+1) certificate bounds that block's
    pair-bag loss by L_t; the staged rule answers 1 - prod_t (1 - L_t).
    This value is *sound* but need not be tight (see check_composition).
    """
    if len(sides) < 2:
        raise ValueError("the staged rule needs at least one block (F >= 2)")
    blocks = []
    for t in range(len(sides) - 1):
        blocks.append(worst_case_loss([sides[t], sides[t + 1]]))
    prod = Fraction(1)
    for bl in blocks:
        prod *= (1 - bl)
    return 1 - prod


def max_discards(specs, allowance: Fraction, mode: str):
    """Largest total discard count whose mask is certified at `allowance`.

    mode="joint"  : the F-factor chain certificate.
    mode="staged" : union-bound composition of per-block certificates.
    """
    best, best_d = -1, None
    for discs in product(*[range(s[3] + 1) for s in specs]):
        sides = [Side(s[0], s[1], s[2], s[3], d) for s, d in zip(specs, discs)]
        try:
            loss = worst_case_loss(sides) if mode == "joint" else staged_loss(sides)
        except InconsistentEvidence:
            continue
        if loss <= allowance:
            total = sum(discs)
            if total > best:
                best, best_d = total, discs
    return best, best_d


def best_mask_allowance(sides_spec, joint: bool):
    """Deprecated helper kept out of the reported checks; see max_discards."""
    raise NotImplementedError("use max_discards(specs, allowance, mode)")


# --------------------------------------------------------------------------
# validations
# --------------------------------------------------------------------------

def _rand_spec(rng, u_max: int, h_max: int, d_max: bool = True):
    """A valid (lo,hi,h,u,d) tuple: hi >= h, lo <= h+u, 0 <= d <= u."""
    u = rng.randint(0, u_max)
    h = rng.randint(0, h_max)
    lo = rng.randint(0, h + u)
    hi = rng.randint(max(lo, h), h + u)
    d = rng.randint(0, u) if d_max else 0
    return (lo, hi, h, u, d)


def _rand_nontrivial_spec(rng, u_max: int, h_max: int):
    """A spec with u >= 1, hi >= max(1, h) and h + u >= 1.

    This excludes empty relations and forced-zero intervals, so any observed
    strict gain cannot be blamed on a degenerate block of zero true mass.
    """
    while True:
        u = rng.randint(1, u_max)
        h = rng.randint(0, h_max)
        lo = rng.randint(0, h + u)
        hi = rng.randint(max(lo, h, 1), h + u)
        if hi >= 1:
            return (lo, hi, h, u)


def check_frontier_vs_exact(trials: int = 4000, seed: int = 20261017):
    """Internal consistency only: two author paths inside this module.

    This compares the full vertex product with the 2^F frontier; both are
    author code.  The truth check is check_exact_vs_bruteforce, which
    enumerates masks and labelings independently of the polygon machinery.
    """
    rng = random.Random(seed)
    bad, counts = [], {1: 0, 2: 0, 3: 0}
    for _ in range(trials):
        F = rng.choice((1, 2, 3))
        specs = [_rand_spec(rng, 3, 2) for _t in range(F)]
        p = rng.randint(0, 5)
        q = rng.randint(max(1, p), 10)
        try:
            sides = [Side(*s) for s in specs]
            al = Fraction(p, q)
            exact, fr = support_exact(sides, al), support_frontier(sides, al)
        except InconsistentEvidence:
            continue
        counts[F] += 1
        if exact != fr:
            bad.append({"specs": specs, "allowance": [p, q],
                        "exact": exact, "frontier": fr})
    return {"trials": trials, "cases_per_F": counts,
            "mismatch_count": len(bad), "mismatches": bad[:5]}


def _yield_check(specs, discs, rng, bad, trials):
    p, q = rng.choice(((0, 1), (1, 2), (1, 10), (3, 4), (1, 1)))
    sides = [Side(s[0], s[1], s[2], s[3], d) for s, d in zip(specs, discs)]
    try:
        exact = support_exact(sides, Fraction(p, q))
        bf, _ = support_bruteforce(sides, p, q)
    except InconsistentEvidence:
        return
    trials[0] += 1
    if exact != bf:
        bad.append({"specs": specs, "discards": discs, "allowance": [p, q],
                    "exact": exact, "bruteforce": bf})


def check_exact_vs_bruteforce(seed: int = 7):
    rng = random.Random(seed)
    bad, trials = [], [0]
    for _ in range(400):
        F = rng.choice((2, 3))
        specs, budget = [], 0
        for _t in range(F):
            u = rng.randint(0, 2)
            budget += u
            h = rng.randint(0, 1)
            lo = rng.randint(0, h + u)
            hi = rng.randint(max(lo, h), h + u)
            specs.append((lo, hi, h, u, 0))
        if budget > 6:
            continue
        for discs in product(*[range(s[3] + 1) for s in specs]):
            _yield_check(specs, discs, rng, bad, trials)
    return {"cases": trials[0], "bad_count": len(bad), "bad": bad[:5]}


def check_loss_vs_support(trials: int = 500, seed: int = 99):
    """worst_case_loss must equal the support sign threshold exactly."""
    rng = random.Random(seed)
    bad, checked = [], 0
    for _ in range(trials):
        F = rng.choice((1, 2, 3))
        specs = [_rand_spec(rng, 2, 1) for _t in range(F)]
        try:
            sides = [Side(*s) for s in specs]
            loss = worst_case_loss(sides)
        except InconsistentEvidence:
            continue
        checked += 1
        lam = loss
        at = support_exact(sides, lam)
        if at > 0:
            bad.append({"specs": specs, "loss": str(loss), "support_at_loss": at})
        below = Fraction(lam.numerator * 9 - 1, lam.denominator * 9)
        if below >= 0 and support_exact(sides, below) <= 0:
            bad.append({"specs": specs, "loss": str(loss),
                        "support_below": support_exact(sides, below)})
    return {"checked": checked, "bad_count": len(bad), "bad": bad[:5]}


def check_composition(trials: int = 120, u_max: int = 2, seed: int = 11,
                      nontrivial: bool = False):
    """Soundness and strict gain of the joint chain certificate.

    Soundness claim: for every mask, loss_joint <= loss_staged.
    Gain claim: for some masks loss_joint < loss_staged, so the joint
    certificate certifies masks that the staged rule rejects.
    """
    rng = random.Random(seed)
    gen = _rand_nontrivial_spec if nontrivial else (
        lambda r, um, hm: _rand_spec(r, um, hm, d_max=False)[:4])
    optimistic, strict = [], []
    max_gap, witness = Fraction(0), None
    masks = 0
    for _ in range(trials):
        specs = [gen(rng, u_max, 1) for _t in range(3)]
        for discs in product(*[range(s[3] + 1) for s in specs]):
            sides = [Side(s[0], s[1], s[2], s[3], d) for s, d in zip(specs, discs)]
            try:
                lj, ls = worst_case_loss(sides), staged_loss(sides)
            except InconsistentEvidence:
                continue
            masks += 1
            if lj > ls:
                optimistic.append({"specs": specs, "discards": list(discs),
                                   "joint": str(lj), "staged": str(ls)})
            elif lj < ls:
                strict.append({"specs": specs, "discards": list(discs),
                               "joint": str(lj), "staged": str(ls)})
                if ls - lj > max_gap:
                    max_gap, witness = ls - lj, strict[-1]
    return {"instances": trials, "masks_checked": masks,
            "staged_optimistic_count": len(optimistic),
            "staged_optimistic": optimistic[:3],
            "joint_strictly_tighter_count": len(strict),
            "max_allowance_gain": str(max_gap),
            "gain_witness": witness}


def check_discard_gain(trials: int = 120, u_max: int = 2, seed: int = 5,
                       nontrivial: bool = False):
    """Operational comparison: certified discard count at a fixed allowance."""
    rng = random.Random(seed)
    gen = _rand_nontrivial_spec if nontrivial else (
        lambda r, um, hm: _rand_spec(r, um, hm, d_max=False)[:4])
    out = {}
    for lam in (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2)):
        gains, best_extra, witness = 0, 0, None
        for _ in range(trials):
            specs = [gen(rng, u_max, 1) for _t in range(3)]
            try:
                dj, _ = max_discards(specs, lam, "joint")
                ds, _ = max_discards(specs, lam, "staged")
            except InconsistentEvidence:
                continue
            if dj > ds:
                gains += 1
                if dj - ds > best_extra:
                    best_extra, witness = dj - ds, {"specs": specs,
                                                    "joint_discards": dj,
                                                    "staged_discards": ds}
        out[str(lam)] = {"instances_with_strict_gain": gains,
                         "max_extra_discards": best_extra,
                         "witness": witness}
    return out


def check_gain_window(lam_list=None, h_max: int = 40, outer: int = 1):
    """Deterministic family isolating the joint-vs-staged threshold.

    Family: two fully acquired outer relations with `outer` known positives
    each, and a middle relation holding h known positives, interval [h,h+1]
    and exactly one unknown, which the mask discards.

    Author closed forms.  The chain's total ranges over {h, h+1} while the
    retained count is h, so the joint worst loss is 1/(h+1) and the joint
    rule certifies the discard iff h >= (1-lam)/lam.  Each block alone has
    loss 1/(h+1), so the staged rule asks 1-(h/(h+1))^2 <= lam, i.e.
    h/(h+1) >= sqrt(1-lam).  Both thresholds are confirmed here by exact
    enumeration rather than assumed.
    """
    if lam_list is None:
        lam_list = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2))
    out = {}
    for lam in lam_list:
        p, q = lam.numerator, lam.denominator
        closed_joint = -(-(q - p) // p)          # ceil((1-lam)/lam)
        closed_staged = next(h for h in range(1, 10 ** 6)
                             if h * h * q >= (q - p) * (h + 1) * (h + 1))
        seen = {"joint": None, "staged": None}
        for h in range(1, h_max + 1):
            specs = [(outer, outer, outer, 0), (h, h + 1, h, 1),
                     (outer, outer, outer, 0)]
            for mode in ("joint", "staged"):
                d, _ = max_discards(specs, lam, mode)
                if d >= 1 and seen[mode] is None:
                    seen[mode] = h
        out[str(lam)] = {
            "closed_form_joint_min_h": closed_joint,
            "closed_form_staged_min_h": closed_staged,
            "enumerated_joint_min_h": seen["joint"],
            "enumerated_staged_min_h": seen["staged"],
            "joint_wins_on_h_window": None if (seen["staged"] is None
                                               or seen["joint"] is None)
            else [seen["joint"], seen["staged"] - 1],
            "h_max_scanned": h_max,
        }
    return out


def main() -> int:
    t0 = perf_counter()
    out = {
        "claim1_frontier_vs_exact": check_frontier_vs_exact(),
        "claim1_exact_vs_bruteforce": check_exact_vs_bruteforce(),
        "claim1_loss_vs_support": check_loss_vs_support(),
        "claim3_composition": check_composition(),
        "claim3_composition_nontrivial": check_composition(
            trials=200, u_max=2, seed=77, nontrivial=True),
        "claim4_discard_gain": check_discard_gain(),
        "claim4_discard_gain_nontrivial": check_discard_gain(
            trials=200, u_max=2, seed=515, nontrivial=True),
        "claim4_gain_window": check_gain_window(),
    }
    out["seconds"] = round(perf_counter() - t0, 3)
    print(json.dumps(out, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
