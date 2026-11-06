# -*- coding: utf-8 -*-
"""Matched-triplet verification: one harness, three planners.

Proves that the evidence chain
    per-side (marginal) 232,714 -> joint+greedy 198,142 -> joint+exact 197,563
differs ONLY in the certificate/allocation policy.  All three runs share,
by construction, the exact same:
  - audit: make_regions(case, oracle, seed, 500) intervals + acquired facts
  - execution strategy: degree-ordered acquisition, batch 128, dynamic
    partner pruning (possible_unknowns), forced-discard containment
  - charging: complete oracle requests (audit + execution) from oracle.ledger

The only swapped component is the planner callback:
  (a) marginal: per-side certified caps, best-case split, fixed at the
      initial evidence (same bag guarantee 0.9)
  (b) greedy  : heap allocation over the same exact four-pair profiles
  (c) exact   : min-plus all-mask optimum (count_mask_planner.plan_counts)

Expected pooled requests (budget 500, lambda=0.1, 48 cells):
  marginal 232,714 / greedy 198,142 / exact 197,563  (asserted per policy)
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
from count_mask_planner import SideCounts, plan_counts
import count_mask_planner as exact_mod
from greedy_planner import allocate_profiles
from marginal_full_loop import best_split
from marginal_baseline import a_max

LAM = Fraction(1, 10)
BUDGET = 500
SEEDS = [101, 202, 303]


def side_metadata(case, intervals, facts, ids, forced=0):
    unknown = [j for j in ids if j not in facts]
    h = int(sum(facts.get(j) == 1 for j in ids))
    lo, hi = intervals.get(tuple(ids), (0, 0))
    return SideCounts(lo, hi, h, len(unknown), forced), unknown


def planner_marginal(meta, splits, target=0.9):
    """Per-side caps; split chosen once per key at the first plan."""
    counts = []
    for i, (left, right) in enumerate(meta):
        if i not in splits:
            pl = (left.lower, left.upper, left.positive, left.unknown)
            pr = (right.lower, right.upper, right.positive, right.unknown)
            splits[i] = best_split(pl, pr, target)[:2]
        eL, eR = splits[i]
        cL = a_max(left.lower, left.upper, left.positive, left.unknown, eL)
        cR = a_max(right.lower, right.upper, right.positive, right.unknown, eR)
        counts.append((cL, cR))
    return counts


def planner_greedy(meta):
    profiles = tuple(exact_mod._python_profile(a, b, LAM) for a, b in meta)
    ans = allocate_profiles(profiles)
    if ans is None:
        raise InconsistentEvidence("greedy: no certified superset")
    return [tuple(c) for c in ans.counts]


def planner_exact(meta):
    ans = plan_counts(meta, LAM)
    if ans is None:
        raise InconsistentEvidence("exact: no certified superset")
    return [tuple(c) for c in ans.counts]


def run_policy(case, seed, budget, planner):
    oracle = Oracle(case)
    intervals, _, _ = make_regions(case, oracle, seed, budget)
    incident = set(j for pair in zip(case.key_left, case.key_right) for ids in pair for j in ids)
    rank = np.ones(case.n) * potential_degrees(case, {}, opposite_scores=np.ones(case.n))
    execution_order = np.lexsort((np.arange(case.n), -rank))
    splits = {}

    def plan(facts, old):
        old = set(old) - facts.keys()
        mask = (set(range(case.n)) - incident - facts.keys()) | old
        meta = []
        for left_ids, right_ids in zip(case.key_left, case.key_right):
            lc, lu = side_metadata(case, intervals, facts, left_ids, len(old.intersection(left_ids)))
            rc, ru = side_metadata(case, intervals, facts, right_ids, len(old.intersection(right_ids)))
            meta.append((lc, rc))
        counts = planner(meta, splits) if planner is planner_marginal else planner(meta)
        for pair, (lids, rids) in zip(counts, zip(case.key_left, case.key_right)):
            for ids, count in zip((lids, rids), pair):
                unknown = [j for j in ids if j not in facts]
                forced = old.intersection(unknown)
                if count < len(forced):
                    raise InconsistentEvidence("cap below forced discards")
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
    print(f"eligible: {len(eligible)}")

    rows = []
    for case in eligible:
        for seed in SEEDS:
            row = {"case_id": case.case_id, "seed": seed}
            for name, planner in (("marginal", planner_marginal),
                                  ("greedy", planner_greedy),
                                  ("exact", planner_exact)):
                r = run_policy(case, seed, BUDGET, planner)
                row[f"{name}_requests"] = r["requests"]
                row[f"{name}_conflict"] = r["conflict"]
                print(f"{case.case_id[:38]:40s} s{seed} {name:8s} requests={r['requests']} "
                      f"conflict={r['conflict']}", flush=True)
            rows.append(row)

    out = HERE / "matched_triplet_results.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    for name in ("marginal", "greedy", "exact"):
        tot = sum(r[f"{name}_requests"] for r in rows)
        cf = sum(1 for r in rows if r[f"{name}_conflict"])
        print(f"{name:8s} pooled {tot:,}  conflicts {cf}/48")
    M = sum(r["marginal_requests"] for r in rows)
    G = sum(r["greedy_requests"] for r in rows)
    E = sum(r["exact_requests"] for r in rows)
    print(f"marginal->greedy saving {(M - G) / M * 100:.2f}%  "
          f"greedy->exact {(G - E) / G * 100:.2f}%  "
          f"marginal->exact {(M - E) / M * 100:.2f}%")
    assert M == 232714 and G == 198142 and E == 197563, (M, G, E)
    print("MATCHED-TRIPLET REPRODUCES THE PAPER NUMBERS EXACTLY")


if __name__ == "__main__":
    main()
