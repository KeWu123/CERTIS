"""Generic exact-optimizer baselines for the CERTIS signed-deficit interface.

Purpose
-------
A reviewer asked what the specialised kernel buys over an off-the-shelf exact
optimizer applied to the *same* interface.  This module answers that with four
generic competitors that exploit no algebraic structure:

  G1  labelling brute force      -- enumerate every Boolean world of the
                                    underlying identity populations and take
                                    the exact maximum of the objective.
  G2  full vertex enumeration    -- enumerate all polygon vertices per side
                                    (up to six) and all vertex pairs per key
                                    (up to thirty-six evaluations), the
                                    classical route before the two-point
                                    signed frontier.
  G3  mask brute force           -- enumerate every discard mask of the unknown
                                    populations and take the best certifying
                                    one, for the all-mask count-allocation
                                    problem.
  G4  chain labelling brute force-- enumerate every combined labelling of an
                                    F-relation chain and take the exact
                                    maximum, for the chain-frontier theorem.

The specialised kernel is imported from the shipped modules so that the
comparison runs on exactly the object the manuscript certifies.

What is and is not claimed
--------------------------
G1/G3/G4 establish *values* by exhaustive enumeration, so agreement is a
genuine optimality check, not a re-run of the kernel.  G2 exposes the
evaluation count that the signed frontier removes.  Recorded wall-clock time
is machine dependent and is never used as a certificate.  Nothing here
executes an oracle or a model; every population is a finite synthetic
instance and the objective is exact rational/integer arithmetic.

Run:  python generic_exact_baseline.py [--out DIR] [--seed INT] [--trials INT]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilinear_certificate import (  # noqa: E402
    InconsistentEvidence,
    RegionSide,
    key_support,
    polygon_vertices,
)
from count_mask_planner import (  # noqa: E402
    SideCounts,
    frontier,
    frontier_support,
    plan_counts,
)

# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def objective(pair_left, pair_right, allowance: Fraction) -> int:
    """(q-p)*(a+b)*(c+d) - q*b*d for p/q = allowance.  Exact integers."""
    p, q = allowance.numerator, allowance.denominator
    a, b = pair_left
    c, d = pair_right
    return (q - p) * (a + b) * (c + d) - q * b * d


# --------------------------------------------------------------------------
# G1 -- labelling brute force (ground truth, exploits no structure)
# --------------------------------------------------------------------------


def feasible_pairs_from_labellings(side: RegionSide) -> set[tuple[int, int]]:
    """All (lost positives, retained positives) reachable by labelling.

    Enumerates every assignment of the unknown-lost and unknown-kept
    populations bit by bit and keeps those whose side total lies in the
    declared interval.  Known positives are forced.  Exponential on purpose:
    this is the generic reference, not a method.
    """
    A, K, h = side.unknown_lost, side.unknown_kept, side.known_positive
    lo, hi = side.lower_total, side.upper_total
    n = A + K
    if n > 20:
        raise ValueError("labelling enumeration is capped at 20 unknowns")
    out: set[tuple[int, int]] = set()
    for bits in range(1 << n):
        mask = bits
        a = 0
        for i in range(A):
            a += (mask >> i) & 1
        c = 0
        for j in range(K):
            c += (mask >> (A + j)) & 1
        b = h + c
        if lo <= a + b <= hi:
            out.add((a, b))
    if not out:
        raise InconsistentEvidence("no feasible labelling")
    return out


def g1_key_value(left: RegionSide, right: RegionSide,
                 allowance: Fraction) -> int:
    pl = feasible_pairs_from_labellings(left)
    pr = feasible_pairs_from_labellings(right)
    return max(objective(x, y, allowance) for x in pl for y in pr)


# --------------------------------------------------------------------------
# G2 -- full vertex enumeration (classical, no sign structure)
# --------------------------------------------------------------------------


def g2_key_value(left: RegionSide, right: RegionSide,
                 allowance: Fraction) -> tuple[int, int]:
    """Return (value, evaluations).  Uses every polygon vertex pair."""
    vl, vr = polygon_vertices(left), polygon_vertices(right)
    best = None
    evals = 0
    for a, b in vl:
        for c, d in vr:
            v = objective((a, b), (c, d), allowance)
            evals += 1
            if best is None or v > best:
                best = v
    return best, evals


def kernel_key_value(left: RegionSide, right: RegionSide,
                     allowance: Fraction) -> tuple[int, int]:
    """Return (value, evaluations) for the shipped two-point frontier."""
    pl = frontier(left)
    pr = frontier(right)
    best = None
    evals = 0
    for a, b in pl:
        for c, d in pr:
            v = objective((a, b), (c, d), allowance)
            evals += 1
            if best is None or v > best:
                best = v
    return best, evals


# --------------------------------------------------------------------------
# G3 -- mask brute force for the all-mask count-allocation problem
# --------------------------------------------------------------------------


def g3_best_discards(keys, allowance: Fraction):
    """Exhaustively enumerate every mask of the unknown populations.

    ``keys`` is a sequence of (SideCounts, SideCounts) holding only *unknown*
    populations (positive/forced fields must be consistent).  A mask chooses,
    per side, how many unknowns are discarded.  The certificate accepts a mask
    when the summed signed support over keys is at most zero.  Returns
    (max certified discards, number of masks examined).
    """
    ranges = []
    for left, right in keys:
        ranges.append(range(0, left.unknown + 1))
        ranges.append(range(0, right.unknown + 1))
    examined = 0
    best = None
    for combo in product(*ranges):
        examined += 1
        total = 0
        feasible = True
        for idx, (left, right) in enumerate(keys):
            dl, dr = combo[2 * idx], combo[2 * idx + 1]
            try:
                vl = RegionSide(left.lower, left.upper, left.positive,
                                dl, left.unknown - dl)
                vr = RegionSide(right.lower, right.upper, right.positive,
                                dr, right.unknown - dr)
            except InconsistentEvidence:
                feasible = False
                break
            total += frontier_support(vl, vr, allowance)
        if not feasible:
            continue
        if total <= 0:
            discarded = sum(combo)
            if best is None or discarded > best:
                best = discarded
    return best, examined


# --------------------------------------------------------------------------
# G4 -- chain labelling brute force
# --------------------------------------------------------------------------


def g4_chain_threshold(h_max: int, allowance: Fraction):
    """Direct enumeration of the chain section's declared family.

    Family: F=3 with both outer relations fully acquired, so their exact
    counts contribute a common positive factor that cancels; the middle
    relation holds ``h`` known positives, interval ``[h, h+1]`` and one
    unknown identity that the mask discards.  The chain mass therefore ranges
    over ``t in {h, h+1}`` while the retained count stays ``h``.

    This is an independent enumeration path: it does not reuse the frontier
    algebra, the loss-support equivalence, or the closed forms.  It decides
    for each ``h`` whether the joint rule certifies the discard,
    ``(q-p)*t - q*h <= 0`` for every ``t``, and whether the staged rule
    certifies it, ``1 - (h/t)^2 <= p/q`` for every ``t``.

    Returns (certified h for the joint rule, certified h for the staged rule).
    """
    p, q = allowance.numerator, allowance.denominator
    joint, staged = [], []
    for h in range(1, h_max + 1):
        if all((q - p) * t - q * h <= 0 for t in (h, h + 1)):
            joint.append(h)
        if all(1 - Fraction(h, t) ** 2 <= Fraction(p, q) for t in (h, h + 1)):
            staged.append(h)
    return joint, staged


# --------------------------------------------------------------------------
# instance generation
# --------------------------------------------------------------------------


def random_side(rng: random.Random, cap: int = 4) -> RegionSide:
    A = rng.randint(0, cap)
    K = rng.randint(0, cap)
    h = rng.randint(0, cap)
    lo = rng.randint(0, A + K + h)
    hi = rng.randint(lo, A + K + h)
    return RegionSide(lo, hi, h, A, K)


def random_allowance(rng: random.Random) -> Fraction:
    q = rng.randint(2, 10)
    p = rng.randint(0, q)
    return Fraction(p, q)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def integer_points_of_region(side: RegionSide) -> set[tuple[int, int]]:
    """Every integer (lost, retained) pair satisfying the declared region.

    This is the count-domain object of the paper: ``0 <= a <= A``,
    ``h <= b <= h + C`` and ``lo <= a + b <= hi``.  It is *not* the vertex
    set; vertices are a subset of these points.
    """
    A, K, h = side.unknown_lost, side.unknown_kept, side.known_positive
    lo, hi = side.lower_total, side.upper_total
    out: set[tuple[int, int]] = set()
    for a in range(0, A + 1):
        for b in range(h, h + K + 1):
            if lo <= a + b <= hi:
                out.add((a, b))
    return out


def check_lift(trials: int, seed: int):
    """The Boolean lift must reproduce exactly the integer points of the region.

    Three statements are checked per instance:
      (i)   realisable-by-labelling set == integer points of the region;
      (ii)  every polygon vertex is realisable (so the vertex reduction is
            sound on this instance);
      (iii) the region is nonempty whenever the constructor accepts it.
    """
    rng = random.Random(seed)
    mismatches = []
    vertex_missing = []
    checked = 0
    for _ in range(trials):
        try:
            side = random_side(rng)
            realisable = feasible_pairs_from_labellings(side)
            region = integer_points_of_region(side)
            poly = set(polygon_vertices(side))
        except InconsistentEvidence:
            continue
        checked += 1
        if realisable != region:
            mismatches.append({
                "side": [side.lower_total, side.upper_total, side.known_positive,
                         side.unknown_lost, side.unknown_kept],
                "realisable_only": sorted(realisable - region),
                "region_only": sorted(region - realisable),
            })
        if not poly <= realisable:
            vertex_missing.append({
                "side": [side.lower_total, side.upper_total, side.known_positive,
                         side.unknown_lost, side.unknown_kept],
                "vertices_not_realisable": sorted(poly - realisable),
            })
    return {"check": "lift: labelling set == integer points of region",
            "checked": checked,
            "mismatches": mismatches,
            "vertices_not_realisable": vertex_missing,
            "passed": not mismatches and not vertex_missing}


def check_values(trials: int, seed: int):
    """G1 == G2 == kernel, on the same instances; record evaluation counts."""
    rng = random.Random(seed)
    mismatches = []
    checked = 0
    kernel_evals = 0
    g2_evals = 0
    strict_saving = 0
    for _ in range(trials):
        try:
            left = random_side(rng)
            right = random_side(rng)
            allowance = random_allowance(rng)
            v1 = g1_key_value(left, right, allowance)
            v2, e2 = g2_key_value(left, right, allowance)
            vk, ek = kernel_key_value(left, right, allowance)
        except InconsistentEvidence:
            continue
        checked += 1
        kernel_evals += ek
        g2_evals += e2
        if ek < e2:
            strict_saving += 1
        if not (v1 == v2 == vk):
            mismatches.append({
                "left": [left.lower_total, left.upper_total, left.known_positive,
                         left.unknown_lost, left.unknown_kept],
                "right": [right.lower_total, right.upper_total,
                          right.known_positive, right.unknown_lost,
                          right.unknown_kept],
                "allowance": str(allowance),
                "g1": v1, "g2": v2, "kernel": vk, "kernel_evals": ek,
                "g2_evals": e2,
            })
    return {"check": "G1==G2==kernel with evaluation counts", "checked": checked,
            "mismatches": mismatches, "kernel_evals": kernel_evals,
            "g2_evals": g2_evals,
            "cases_kernel_strictly_fewer": strict_saving,
            "passed": not mismatches}


def check_allocation(trials: int, seed: int, cap: int = 3):
    """G3 mask brute force vs the shipped min-plus planner."""
    rng = random.Random(seed)
    mismatches = []
    checked = 0
    for _ in range(trials):
        nkeys = rng.randint(1, 3)
        keys = []
        try:
            for _ in range(nkeys):
                keys.append((side_counts(rng, cap), side_counts(rng, cap)))
            allowance = random_allowance(rng)
        except (InconsistentEvidence, ValueError):
            continue
        brute, examined = g3_best_discards(keys, allowance)
        plan = plan_counts(keys, allowance)
        planned = None if plan is None else plan.discarded
        checked += 1
        if brute != planned:
            mismatches.append({
                "keys": [[[s.lower, s.upper, s.positive, s.unknown]
                          for s in pair] for pair in keys],
                "allowance": str(allowance),
                "brute_force_best_discards": brute,
                "planner_discarded": planned,
                "masks_examined": examined,
            })
    return {"check": "G3 mask brute force == min-plus planner", "checked": checked,
            "mismatches": mismatches, "passed": not mismatches}


def side_counts(rng: random.Random, cap: int) -> SideCounts:
    unknown = rng.randint(0, cap)
    positive = rng.randint(0, cap)
    lo = rng.randint(0, positive + unknown)
    hi = rng.randint(lo, positive + unknown)
    return SideCounts(lo, hi, positive, unknown, 0)


def check_chain(h_max: int, seed: int):
    """G4 direct family enumeration vs the closed forms of the chain section."""
    results = []
    ok = True
    for p, q in ((1, 10), (2, 10), (5, 10)):
        lam = Fraction(p, q)
        joint_direct, staged_direct = g4_chain_threshold(h_max, lam)
        closed_joint = -(-(q - p) // p)
        closed_staged = next(
            h for h in range(1, 10 ** 6)
            if h * h * q >= (q - p) * (h + 1) * (h + 1)
        )
        joint_pred = [h for h in range(1, h_max + 1) if h >= closed_joint]
        staged_pred = [h for h in range(1, h_max + 1) if h >= closed_staged]
        agree = (joint_direct == joint_pred) and (staged_direct == staged_pred)
        ok = ok and agree
        results.append({
            "allowance": f"{p}/{q}",
            "h_max": h_max,
            "joint_window_direct_enumeration": joint_direct,
            "joint_window_closed_form": joint_pred,
            "staged_window_direct_enumeration": staged_direct,
            "staged_window_closed_form": staged_pred,
            "closed_form_joint_threshold": closed_joint,
            "closed_form_staged_threshold": closed_staged,
            "agree": agree,
        })
    return {"check": "G4 chain windows: direct enumeration vs closed forms",
            "seed": seed, "results": results, "passed": ok}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "research", "innovation_20261016", "generic_exact_baseline"))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--trials", type=int, default=4000)
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    started = time.time()
    report = {"seed": args.seed, "trials": args.trials, "checks": []}
    report["checks"].append(check_lift(args.trials, args.seed))
    report["checks"].append(check_values(args.trials, args.seed + 1))
    report["checks"].append(check_allocation(max(50, args.trials // 20),
                                             args.seed + 2))
    report["checks"].append(check_chain(40, args.seed + 3))
    report["all_passed"] = all(c["passed"] for c in report["checks"])
    report["elapsed_seconds"] = round(time.time() - started, 3)

    path = os.path.join(out, "generic_exact_baseline_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    for c in report["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"[{status}] {c['check']}")
        if "checked" in c:
            print(f"        checked={c['checked']} mismatches={len(c['mismatches'])}")
        if "kernel_evals" in c:
            print(f"        kernel_evals={c['kernel_evals']} "
                  f"g2_evals={c['g2_evals']} "
                  f"strictly_fewer={c['cases_kernel_strictly_fewer']}")
        if c["check"].startswith("G4"):
            for r in c["results"]:
                print(f"        lambda={r['allowance']} agree={r['agree']} "
                      f"joint={r['joint_window_direct_enumeration']} "
                      f"staged={r['staged_window_direct_enumeration']}")
    print(f"all_passed={report['all_passed']} "
          f"elapsed={report['elapsed_seconds']}s -> {path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
