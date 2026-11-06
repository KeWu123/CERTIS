# -*- coding: utf-8 -*-
"""Full feedback execution under MARGINAL (per-side) certification.

The marginal policy mirrors run_all_mask mechanically (same audits, ranking,
batches, partner pruning, forced-discard containment, id-order additions),
but replaces the joint all-mask count plan with per-side certified caps:

  for each key, each side:  discard <= A_max(lo, hi, h, n_unknown, eps_side)
  with the best-case asymmetric split maximizing A_L + A_R subject to
  (1-eps_L)(1-eps_R) >= bag_target  (bag recall >= target per key).

Two targets:
  target=0.9  -> matches the joint certificate's bag guarantee 0.9
  target=0.81 -> matches the joint ALLOWANCE 0.1 (per-side 0.1 each)

Read-only replay of the archived protocol tapes; no new oracle calls beyond
the loop itself (same acquisition order as the archived all-mask runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ES = HERE.parent / "external_study"
CODE = HERE.parents[2] / "code"
sys.path.insert(0, str(ES))
sys.path.insert(0, str(CODE))

from common import load_cases, Oracle, potential_degrees, possible_unknowns
from direct_arms import make_regions
from bilinear_certificate import InconsistentEvidence

from marginal_baseline import a_max, EPS_GRID

BUDGET = 500
SEEDS = [101, 202, 303]
TARGETS = [0.9, 0.81]


def best_split(pl, pr, target):
    """Best-case asymmetric split for one key: return (eps_L, eps_R, total)."""
    best = (0.0, 0.0, 0)
    for eL in EPS_GRID:
        if eL > 1.0 - target / 0.995:  # beyond this even eps_R=0.005 fails
            break
        eR = 1.0 - target / (1.0 - eL)
        if eR < 0.0049 or eR > 0.4:
            continue
        total = a_max(*pl, eL) + a_max(*pr, eR)
        if total > best[2]:
            best = (eL, eR, total)
    return best


def run_marginal(case, seed, budget, target):
    oracle = Oracle(case)
    intervals, _, _ = make_regions(case, oracle, seed, budget)
    incident = set(j for pair in zip(case.key_left, case.key_right) for ids in pair for j in ids)
    rank = np.ones(case.n) * potential_degrees(case, {}, opposite_scores=np.ones(case.n))
    execution_order = np.lexsort((np.arange(case.n), -rank))

    def plan(facts, old):
        old = set(old) - facts.keys()
        mask = (set(range(case.n)) - incident - facts.keys()) | old
        for left_ids, right_ids in zip(case.key_left, case.key_right):
            pl = pr = None
            for ids, which in ((left_ids, 0), (right_ids, 1)):
                lo, hi = intervals.get(tuple(ids), (0, 0))
                h = int(sum(facts.get(j) == 1 for j in ids))
                n_unknown = len([j for j in ids if j not in facts])
                if which == 0:
                    pl = (lo, hi, h, n_unknown)
                else:
                    pr = (lo, hi, h, n_unknown)
            key = tuple(left_ids), tuple(right_ids)
            if key not in splits:
                # Split is chosen ONCE, at the first plan, and then fixed:
                # re-optimising it per round can drop a side's cap below its
                # forced discards, which the containment protocol forbids.
                splits[key] = best_split(pl, pr, target)[:2]
            eL, eR = splits[key]
            for ids, params, e in ((left_ids, pl, eL), (right_ids, pr, eR)):
                cap = a_max(*params, e)
                unknown = [j for j in ids if j not in facts]
                forced = old.intersection(unknown)
                if cap < len(forced):
                    raise InconsistentEvidence("marginal cap below forced discards")
                mask.update(forced)
                additions = [j for j in unknown if j not in forced][:cap - len(forced)]
                mask.update(additions)
        return mask

    splits = {}
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
    return {"unknown_discards": len(mask), "requests": len(oracle.ledger),
            "conflict": conflict,
            "tokens_in": int(sum(r["tokens_in"] for r in oracle.ledger)),
            "tokens_out": int(sum(r["tokens_out"] for r in oracle.ledger))}


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
    print(f"eligible single-key cases: {len(eligible)}", flush=True)

    rows = []
    for case in eligible:
        for seed in SEEDS:
            for target in TARGETS:
                m = run_marginal(case, seed, BUDGET, target)
                rows.append({
                    "case_id": case.case_id, "seed": seed, "budget": BUDGET,
                    "target": target,
                    "marg_requests": m["requests"],
                    "marg_discard": m["unknown_discards"],
                    "marg_conflict": m["conflict"],
                    "tokens_in": m["tokens_in"],
                    "tokens_out": m["tokens_out"],
                })
                print(f"{case.case_id[:40]} s{seed} t{target}: "
                      f"requests={m['requests']} discard={m['unknown_discards']} "
                      f"conflict={m['conflict']}", flush=True)
    out = HERE / "marginal_full_loop_results.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
