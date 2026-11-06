# -*- coding: utf-8 -*-
"""Mechanism decomposition row B: global signed-support budget with
SEPARATE extrema (loss maximum and answer minimum computed independently),
greedy-count allocation, on the same 48 matched cells.

  A. best per-side certification                    232,714 (matched_triplet)
  B. global budget + separate extrema + greedy       <- this script
  C. four-pair joint certificate + greedy-count     198,142 (matched_triplet)
  D. four-pair joint certificate + exact DP         197,563 (matched_triplet)

Separate-extrema per-key support at discard counts (d_L, d_R):
  H^sep_k = (q-p) * tmax_L(d_L) * tmax_R(d_R) - q * h_L * h_R
  tmax_s(d) = min(hi_s, h_s + d + k_s)      (max feasible side total)
  B_min = h_L * h_R                          (all known positives retained)
where the loss maximum and the answer minimum may be attained by different
labelings.  H^sep >= H (four-pair coupled), so this profile is strictly
more conservative than the exact profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np

HERE = Path(__file__).resolve().parent
ES = HERE.parent / "external_study"
GP = HERE.parent / "greedy_count_profile"
CODE = HERE.parents[2] / "code"
sys.path.insert(0, str(ES))
sys.path.insert(0, str(GP))
sys.path.insert(0, str(CODE))

from common import load_cases, Oracle, potential_degrees, possible_unknowns
from direct_arms import make_regions
from bilinear_certificate import InconsistentEvidence
from count_mask_planner import CountProfile, SideCounts
import count_mask_planner as exact_mod
from greedy_planner import allocate_profiles

LAM = Fraction(1, 10)
BUDGET = 500
SEEDS = [101, 202, 303]


def sep_profile(side_l, side_r, allowance):
    """Per-key separate-extrema profile (paper's mechanism-control form).

    H^sep_k(d_L, d_R) = q * lost_max(d_L, d_R) - p * tmin_k,
      lost_max = max over the four-pair candidates of (T - B) at allowance 0
                 (the lost maximum, coupled kernel, Fraction(0))
      tmin_k   = max(lo_L, h_L) * max(lo_R, h_R)   (d-independent answer minimum)
    The loss maximum and the answer minimum may come from different
    labelings, so H^sep >= H: strictly conservative.

    Reuses the exact profile kernel at allowance 0 for the lost component;
    returns a CountProfile for the greedy allocator.
    """
    p, q = allowance.numerator, allowance.denominator
    lost = exact_mod._python_profile(side_l, side_r, Fraction(0))
    tmin = (max(side_l.lower, side_l.positive)
            * max(side_r.lower, side_r.positive))
    values = tuple(q * v - p * tmin for v in lost.values)
    return CountProfile(lost.forced, values, lost.choices)


class SepGreedyPlan:
    def __init__(self, counts, support):
        self.counts = counts
        self.support = support


def sep_greedy_plan(meta):
    """Greedy heap allocation over separate-extrema profiles, global budget."""
    profiles = []
    for (sl, sr) in meta:
        profiles.append(sep_profile(sl, sr, LAM))
    ans = allocate_profiles(profiles)
    if ans is None:
        raise InconsistentEvidence("sep-greedy: no certified superset")
    return SepGreedyPlan(ans.counts, ans.support)


def run_sep_greedy(case, seed, budget):
    oracle = Oracle(case)
    intervals, _, _ = make_regions(case, oracle, seed, budget)
    incident = set(j for pair in zip(case.key_left, case.key_right) for ids in pair for j in ids)
    rank = np.ones(case.n) * potential_degrees(case, {}, opposite_scores=np.ones(case.n))
    execution_order = np.lexsort((np.arange(case.n), -rank))

    def plan(facts, old):
        old = set(old) - facts.keys()
        mask = (set(range(case.n)) - incident - facts.keys()) | old
        meta = []
        for left_ids, right_ids in zip(case.key_left, case.key_right):
            pair = []
            for ids in (left_ids, right_ids):
                unknown = [j for j in ids if j not in facts]
                h = int(sum(facts.get(j) == 1 for j in ids))
                lo, hi = intervals.get(tuple(ids), (0, 0))
                forced = len(old.intersection(unknown))
                pair.append(SideCounts(lo, hi, h, len(unknown), forced))
            meta.append(tuple(pair))
        planobj = sep_greedy_plan(meta)
        for pair, split in zip(zip(case.key_left, case.key_right), planobj.counts):
            for ids, count in zip(pair, split):
                unknown = [j for j in ids if j not in facts]
                forced = old.intersection(unknown)
                if count < len(forced):
                    raise InconsistentEvidence("sep cap below forced")
                mask.update(forced)
                mask.update([j for j in unknown if j not in forced][:count - len(forced)])
        return mask

    mask = plan(oracle.facts, set())
    conflict = False
    while True:
        retained = np.ones(case.n, dtype=bool)
        retained[list(mask)] = False
        possible = set(possible_unknowns(case, oracle.facts, retained))
        selected = [int(v) for v in execution_order if int(v) in possible][:128]
        if not selected:
            break
        oracle.acquire(selected, "execution")
        if not conflict:
            try:
                mask = plan(oracle.facts, mask)
            except InconsistentEvidence:
                conflict = True
    return {"requests": len(oracle.ledger), "discards": len(mask), "conflict": conflict}


def main():
    cases = [c for c in load_cases() if c.variant == "single"]
    eligible = []
    for c in cases:
        mem = np.zeros(c.n, dtype=np.int64)
        for a, b in zip(c.key_left, c.key_right):
            mem[list(a)] += 1
            mem[list(b)] += 1
        if bool(np.all(mem <= 1)):
            eligible.append(c)

    rows = []
    for case in eligible:
        for seed in SEEDS:
            r = run_sep_greedy(case, seed, BUDGET)
            rows.append({"case_id": case.case_id, "seed": seed, "requests": r["requests"],
                         "conflict": r["conflict"]})
            print(f"{case.case_id[:38]:40s} s{seed} requests={r['requests']} "
                  f"conflict={r['conflict']}", flush=True)
    out = HERE / "sep_extrema_results.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tot = sum(r["requests"] for r in rows)
    cf = sum(1 for r in rows if r["conflict"])
    print(f"\nsep-extrema greedy pooled {tot:,}  conflicts {cf}/48")
    print("saved", out.name)


if __name__ == "__main__":
    main()
