"""Timer-only adapter equivalence, boundary and event-accounting checks."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import random
import sys

from policy_adapter import run_greedy_count as original
from policy_adapter_v2 import run_greedy_count as timed
from greedy_planner import exact, allocate_profiles, plan_counts
from common import make_tiny_case

HERE = Path(__file__).resolve().parent


def main():
    checked = 0
    for truth in product((0, 1), repeat=6):
        case = make_tiny_case(side=(0,)*3+(1,)*3, key_left=((0, 1), (2,)),
                             key_right=((3, 4), (5,)), truth=truth, name='timer_equivalence')
        a = original(case, 101, 1)
        b = timed(case, 101, 1)
        for field in ('oracle_ledger', 'helper_ledger', 'facts', 'regions', 'discarded',
                      'initial_discarded', 'initial_retained_unknown', 'final_scaled_support',
                      'initial_certified_discard_count', 'final_certified_discard_count'):
            assert a[field] == b[field]
        assert len(b['planner_observations']) == b['projection_calls']
        assert all(o['process_cpu_seconds'] >= 0 and o['wall_seconds'] >= 0 for o in b['planner_observations'])
        assert all(e['chosen_unknown_discard_count'] == e['discard_cardinality'] for e in b['events'])
        assert b['events'][0]['chosen_unknown_discard_count'] == b['initial_certified_discard_count']
        assert b['events'][-1]['chosen_unknown_discard_count'] == b['final_certified_discard_count']
        checked += 1
        case.close_sql()
    # Tie budget one makes the original-key tie break observable.
    profiles = (exact.CountProfile(0, (-1, 0), ((0, 0), (1, 0))),
                exact.CountProfile(0, (0, 1), ((0, 0), (1, 0))))
    assert allocate_profiles(profiles).chosen_extra_counts == (1, 0)
    # Forced old mask can itself be uncertifiable; no safe extension exists.
    forced = exact.SideCounts(1, 1, 0, 1, 1)
    kept = exact.SideCounts(1, 1, 1, 0)
    assert plan_counts(((forced, kept),), Fraction(1, 10)) is None
    # Find a discriminating adverse case using actual exact local profiles,
    # then preserve it as a witness instead of assuming global optimality.
    rng = random.Random(2026101702)
    catalogue = []
    for _ in range(600):
        sides = []
        for _ in range(2):
            u, h = rng.randrange(1, 8), rng.randrange(5)
            lo, hi = sorted((rng.randrange(h+u+1), rng.randrange(h+u+1)))
            sides.append(exact.SideCounts(min(lo, h+u), max(hi, h), h, u))
        profile = exact._python_profile(*sides, Fraction(1, 10))
        catalogue.append((sides, profile))
    witness = None
    for _ in range(30000):
        selected = rng.sample(catalogue, 3)
        greedy = allocate_profiles([p for _, p in selected])
        if greedy is None: continue
        best = max(sum(indices) for indices in product(*(range(len(p.values)) for _, p in selected))
                   if sum(p.values[i] for i, (_, p) in zip(indices, selected)) <= 0)
        if greedy.discarded < best:
            keys = tuple(tuple(sides) for sides, _ in selected)
            dp = exact.plan_counts(keys, Fraction(1, 10), backend='python')
            assert dp.discarded == best and plan_counts(keys, Fraction(1, 10), backend='python').discarded == greedy.discarded
            witness = dict(keys=[[vars(s) for s in sides] for sides, _ in selected],
                           local_values=[p.values for _, p in selected], greedy_discarded=greedy.discarded,
                           exact_dp_discarded=dp.discarded, greedy_chosen_extra=greedy.chosen_extra_counts,
                           exact_dp_counts=dp.counts)
            break
    assert witness is not None, 'Need an actual count-profile adverse witness'
    out = dict(status='pass', timer_adapter_actual_sql_equivalence_cases=checked,
               discriminating_tie_checked=True, forced_noncertifiable_none_checked=True,
               exact_profile_nonconvex_witness=witness,
               source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (HERE/'greedy_planner.py', HERE/'policy_adapter_v2.py', Path(__file__))})
    destination = Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'checks_v2_result.json'
    destination.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
