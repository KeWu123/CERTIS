# -*- coding: utf-8 -*-
"""Joint vs marginal certification, and the three-way pooling decomposition,
all on the PLANNING intervals (make_regions), not the raw ledger regions.

Ledger 'regions' are raw audit records; the archived planner consumed the
coverage-adjusted intervals returned by make_regions.  Re-deriving with the
raw records inflates the per-side caps (the recomputed joint plan would
disagree with the archived optimal_counts).  This script reproduces the
archived initial joint plan exactly (asserted per cell) and computes:

  joint   = archived initial certified discard (events[0])
  perkey  = sum over keys of the per-key two-sided optimum (support_k <= 0)
            -- two-sided structure, no cross-key slack sharing
  marg_*  = per-side certified caps, best-case split (same bag guarantee)

Decomposition at the initial post-audit decision:
  perkey - marg  = value of two-sided (within-key) pooling
  joint - perkey = value of cross-key pooling
"""
import gzip
import glob
import json
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np

HERE = Path(__file__).resolve().parent
ES = HERE.parent / "external_study"
CODE = HERE.parents[2] / "code"
sys.path.insert(0, str(ES))
sys.path.insert(0, str(CODE))

from common import load_cases, Oracle
from direct_arms import make_regions
from bilinear_certificate import RegionSide, key_support
from count_mask_planner import SideCounts, plan_counts

RAW = HERE.parents[2] / "reproduction/archive/primary/count_allocation/raw"

LAM = Fraction(1, 10)
LAM_PRIME = 1.0 - np.sqrt(0.9)
EPS_GRID = np.arange(0.005, 0.4005, 0.005)
SEEDS = [101, 202, 303]
BUDGET = 500


def worst_recall(lo, hi, h, d):
    if hi == 0:
        return 1.0
    if d == 0:
        return 1.0
    best = 1.0
    for T in range(max(lo, h, 1), hi + 1):
        U = T - h
        kept = T - min(U, d)
        r = kept / T
        if r < best:
            best = r
    return best


def a_max(lo, hi, h, n_unknown, lam):
    if hi == 0:
        return n_unknown
    hi_eff = min(hi, h + n_unknown)
    if hi_eff < max(lo, h):
        return 0
    best = 0
    for d in range(0, n_unknown + 1):
        if worst_recall(lo, hi_eff, h, d) >= 1.0 - lam:
            best = d
        else:
            break
    return best


def pair_opt(pl, pr):
    best = 0
    for eL in EPS_GRID:
        if eL > 0.09549:
            break
        eR = 1.0 - 0.9 / (1.0 - eL)
        if eR < 0.0049:
            continue
        total = a_max(*pl, eL) + a_max(*pr, eR)
        if total > best:
            best = total
    return best


def archived_events():
    out = {}
    for f in glob.glob(str(RAW / "*b500*.json.gz")):
        d = json.loads(gzip.decompress(open(f, "rb").read()))
        out[(d["case_id"], d["seed"])] = d["result"]["events"][0]
    return out


def main():
    arch = archived_events()
    assert len(arch) == 48, len(arch)
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
            oracle = Oracle(case)
            intervals, _, _ = make_regions(case, oracle, seed, BUDGET)
            facts = oracle.facts
            ev0 = arch[(case.case_id, seed)]
            joint = ev0["discard_cardinality"]
            # reproduce the archived initial joint plan exactly
            meta = []
            for lid, rid in zip(case.key_left, case.key_right):
                pair = []
                for ids in (lid, rid):
                    unknown = tuple(j for j in ids if j not in facts)
                    h = int(sum(facts.get(j) == 1 for j in ids))
                    lo, hi = intervals.get(tuple(ids), (0, 0))
                    pair.append(SideCounts(lo, hi, h, len(unknown), 0))
                meta.append(tuple(pair))
            ans = plan_counts(meta, LAM)
            assert ans is not None and ans.discarded == joint and ans.support == ev0["scaled_support"], \
                (case.case_id, seed, ans.discarded, joint, ans.support, ev0["scaled_support"])

            perkey = marg_p = marg_1 = marg_opt = 0
            n_pos_keys = 0
            for pair, counts in zip(meta, ans.counts):
                ans1 = plan_counts([pair], LAM)
                assert ans1 is not None
                perkey += ans1.discarded
                pl = (pair[0].lower, pair[0].upper, pair[0].positive, pair[0].unknown)
                pr = (pair[1].lower, pair[1].upper, pair[1].positive, pair[1].unknown)
                marg_p += a_max(*pl, LAM_PRIME) + a_max(*pr, LAM_PRIME)
                marg_1 += a_max(*pl, LAM) + a_max(*pr, LAM)
                marg_opt += pair_opt(pl, pr)
                # per-key signed support of the ARCHIVED joint counts
                sides = []
                for side, count in zip(pair, counts):
                    sides.append(RegionSide(side.lower, side.upper, side.positive,
                                            count, side.unknown - count))
                if key_support(sides[0], sides[1], LAM) > 0:
                    n_pos_keys += 1

            rows.append({"case": case.case_id, "seed": seed,
                         "joint": joint, "perkey": perkey,
                         "marg_p": marg_p, "marg_1": marg_1, "marg_opt": marg_opt,
                         "n_pos_keys": n_pos_keys})
            print(f"{case.case_id[:38]:40s} s{seed} joint={joint:5d} "
                  f"perkey={perkey:5d} marg_opt={marg_opt:5d}", flush=True)

    J = np.array([x["joint"] for x in rows])
    P = np.array([x["perkey"] for x in rows])
    MP = np.array([x["marg_p"] for x in rows])
    M1 = np.array([x["marg_1"] for x in rows])
    MO = np.array([x["marg_opt"] for x in rows])
    print()
    for name, arr in [("joint", J), ("perkey", P), ("marg_p", MP),
                      ("marg_1", M1), ("marg_opt", MO)]:
        print(f"{name:8s} median {np.median(arr):7.1f}  min {arr.min():5d}  max {arr.max():5d}")
    two_sided = P - MO
    cross = J - P
    print(f"two-sided pooling (perkey - marg_opt): median {np.median(two_sided):.1f} "
          f"cells>0 {int(np.sum(two_sided > 0))}/48")
    print(f"cross-key pooling (joint - perkey):    median {np.median(cross):.1f} "
          f"cells>0 {int(np.sum(cross > 0))}/48")
    print(f"joint - marg_opt: min {np.min(J - MO)} median {np.median(J - MO):.1f} max {np.max(J - MO)}")
    print(f"joint - marg_p:   min {np.min(J - MP)} median {np.median(J - MP):.1f} max {np.max(J - MP)}")
    print(f"joint as % of 4000 unknowns: median {np.median(J)/40:.2f}%")
    print(f"marg_opt as %: median {np.median(MO)/40:.2f}%")
    ratio = J / MO
    print(f"joint/marg_opt ratio: median {np.median(ratio):.2f}")
    share_cross = (J - P) / (J - MO)
    print(f"per-cell cross-key share of the gap: median {np.median(share_cross)*100:.1f}% "
          f"min {np.min(share_cross)*100:.1f}%")
    npos = np.array([x["n_pos_keys"] for x in rows])
    print(f"cells with >=1 positive per-key support in the joint plan: {int(np.sum(npos > 0))}/48, "
          f"median {np.median(npos[npos > 0]):.0f} positive keys")
    json.dump(rows, open("marginal_baseline_results.json", "w", encoding="utf-8"), indent=1)
    json.dump(rows, open("perkey_budget_results.json", "w", encoding="utf-8"), indent=1)
    json.dump([{k: r[k] for k in ("case", "seed", "n_pos_keys")} for r in rows],
              open("crosskey_pooling_results.json", "w", encoding="utf-8"), indent=1)
    print("saved marginal_baseline_results.json + perkey_budget_results.json + crosskey_pooling_results.json")


if __name__ == "__main__":
    main()
