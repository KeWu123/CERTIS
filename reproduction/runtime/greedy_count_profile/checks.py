"""Independent finite-world/mask reference, arithmetic and actual SQL checks."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import random
import sys
import time

from greedy_planner import allocate_profiles, plan_counts, exact
from policy_adapter import CountEnvelope, eligible, run_greedy_count, _execution
from common import make_tiny_case
from bilinear_certificate import InconsistentEvidence

HERE = Path(__file__).resolve().parent


def pairs(case, world, mask=()):
    return {(k, i, j) for k, (left, right) in enumerate(zip(case.key_left, case.key_right))
            for i in left for j in right if world[i] and world[j] and i not in mask and j not in mask}


def reference_local(left, right, allowance):
    """Enumerate individual binary labels; no frontier/RegionSide machinery."""
    p, q = allowance.numerator, allowance.denominator
    rows = []
    for dl in range(left.forced_discard, left.unknown + 1):
        for dr in range(right.forced_discard, right.unknown + 1):
            worst = None
            for zl in product((0, 1), repeat=left.unknown):
                lt = left.positive + sum(zl)
                if not left.lower <= lt <= left.upper: continue
                for zr in product((0, 1), repeat=right.unknown):
                    rt = right.positive + sum(zr)
                    if not right.lower <= rt <= right.upper: continue
                    retained = (left.positive + sum(zl[dl:])) * (right.positive + sum(zr[dr:]))
                    value = (q-p)*lt*rt-q*retained
                    worst = value if worst is None else max(worst, value)
            rows.append((dl+dr-left.forced_discard-right.forced_discard, worst, (dl, dr)))
    size = left.unknown + right.unknown-left.forced_discard-right.forced_discard+1
    return tuple(min((v, c) for k, v, c in rows if k == n) for n in range(size))


def main():
    rng = random.Random(2026101701)
    local_checks = mask_checks = sql_checks = adverse = 0
    for repeat in range(120):
        sides = []
        for _ in range(2):
            u, h = rng.randrange(4), rng.randrange(3)
            lo, hi = sorted((rng.randrange(h+u+1), rng.randrange(h+u+1)))
            lo, hi = min(lo, h+u), max(hi, h)
            sides.append(exact.SideCounts(lo, hi, h, u, rng.randrange(u+1)))
        a, b = sides
        allowance = rng.choice((Fraction(0), Fraction(1, 10), Fraction(1, 2), Fraction(1)))
        reference = reference_local(a, b, allowance)
        for backend in ('python', 'numpy'):
            profile = exact._python_profile(a, b, allowance) if backend == 'python' else exact._numpy_profile(a, b, allowance.numerator, allowance.denominator)
            assert tuple(zip(profile.values, profile.choices)) == reference
            assert all(x <= y for x, y in zip(profile.values, profile.values[1:]))
            answer = plan_counts(((a, b),), allowance, backend=backend)
            feasible = [k for k, (v, _) in enumerate(reference) if v <= 0]
            assert (answer is None) == (not feasible)
            if answer: assert answer.discarded == a.forced_discard+b.forced_discard+max(feasible)
            local_checks += 1
    # Synthetic monotone profiles isolate the nonconvex greedy failure.
    # Budget 4: greedy takes key 0's cost 3; key 1's cost 4 then zero would buy 2.
    synthetic = (exact.CountProfile(0, (-4, -1), ((0, 0), (1, 0))),
                 exact.CountProfile(0, (0, 4, 4), ((0, 0), (1, 0), (2, 0))))
    chosen = allocate_profiles(synthetic)
    assert chosen.chosen_extra_counts == (1, 0) and chosen.discarded == 1
    assert max(i+j for i in range(2) for j in range(3) if synthetic[0].values[i]+synthetic[1].values[j] <= 0) == 2
    # A known-only negative term is required to afford the active key increment.
    credit = exact.SideCounts(2, 2, 2, 0)
    unknown = exact.SideCounts(0, 1, 0, 1)
    assert plan_counts(((credit, credit), (unknown, unknown)), Fraction(1, 2)).discarded > 0
    ties = allocate_profiles((exact.CountProfile(0, (-1, 0), ((0, 0), (1, 0))),)*2)
    assert ties.chosen_extra_counts == (1, 1)
    assert plan_counts((), Fraction(1, 10)).discarded == 0
    huge = exact.SideCounts(10**20, 10**20, 10**20, 0)
    assert plan_counts(((huge, huge),), Fraction(1, 10)).backend == 'python-integer'
    try: plan_counts(((huge, huge),), Fraction(1, 10), backend='numpy')
    except OverflowError: pass
    else: raise AssertionError('Unsafe int64 accepted')
    for bad in (0.1, 1, Fraction(-1), Fraction(2)):
        try: plan_counts((), bad)
        except (TypeError, ValueError): pass
        else: raise AssertionError('Invalid allowance accepted')
    # Exhaustive independent all-mask certification also checks forced identities
    # and a free orphan on each side. We compare cardinality without assuming equality.
    for repeat in range(180):
        truth = [rng.randrange(2) for _ in range(8)]
        case = make_tiny_case(side=(0,)*4+(1,)*4, key_left=((0, 1), (2,)),
                             key_right=((4,), (5, 6)), truth=truth, name='greedy_mask')
        facts = {j: y for j, y in enumerate(truth) if rng.randrange(4) == 0}
        intervals = {}
        for ids in (*case.key_left, *case.key_right):
            total = sum(truth[j] for j in ids)
            intervals[ids] = (rng.randrange(total+1), rng.randrange(total, len(ids)+1))
        old = {j for j in range(case.n) if j not in facts and rng.randrange(6) == 0}
        worlds = [z for z in product((0, 1), repeat=8) if all(z[j] == y for j, y in facts.items())
                  and all(lo <= sum(z[j] for j in ids) <= hi for ids, (lo, hi) in intervals.items())]
        feasible = []
        for bits in product((0, 1), repeat=8):
            mask = {j for j, bit in enumerate(bits) if bit}
            if mask & facts.keys() or not old <= mask: continue
            worst = max(9*len(pairs(case, z))-10*len(pairs(case, z, mask)) for z in worlds)
            if worst <= 0: feasible.append(mask)
        envelope = CountEnvelope(case, intervals)
        try: mask, answer = envelope.plan(facts, old)
        except InconsistentEvidence: assert not feasible
        else:
            assert mask in feasible and old <= mask and not mask & facts.keys()
            assert {3, 7}-facts.keys() <= mask
            best = max(map(len, feasible))
            assert len(mask) <= best
            adverse += len(mask) < best
        mask_checks += 1
    # Complete execution, private ledger, actual returned SQL table and cache.
    for truth in product((0, 1), repeat=6):
        case = make_tiny_case(side=(0, 0, 0, 1, 1, 1), key_left=((0, 1), (2,)),
                             key_right=((3, 4), (5,)), truth=truth, name='greedy_sql')
        for audit in (False, True):
            r = run_greedy_count(case, 101, 1, audit=audit)
            actual = case._sql['connection'].execute('SELECT key_id,left_id,right_id FROM returned_derivations').fetchall()
            assert set(actual) == pairs(case, truth, r['discarded'])
            assert len(actual) == r['output']['returned_count']
            assert not r['helper_ledger'] and r['summary']['helper_requests'] == 0
            assert len(r['oracle_ledger']) == len(r['facts']) == r['summary']['oracle_requests']
            assert all('optimal_counts' not in e and 'chosen_counts' in e for e in r['events'])
            sql_checks += 1
        case.close_sql()
    repeated = make_tiny_case(side=(0, 0, 1, 1), key_left=((0, 1), (0,)),
                              key_right=((2,), (3,)), truth=(1, 0, 1, 0), name='repeat')
    assert not eligible(repeated)
    try: CountEnvelope(repeated, {})
    except ValueError: pass
    else: raise AssertionError('Repeated identity accepted')
    # Verify policy injection cannot mutate production's global planner binding.
    import count_allocation_arms as production
    assert production.plan_counts is exact.plan_counts and _execution.plan_counts is plan_counts
    out = dict(status='pass', local_profile_world_enumerations=local_checks,
               independent_tiny_mask_checks=mask_checks, tiny_suboptimal_cardinalities=adverse,
               actual_sql_executions=sql_checks, synthetic_nonconvex_adverse_case=True,
               signed_credit_checked=True, overflow_fallback_checked=True,
               forced_identity_orphans_checked=True, production_binding_unchanged=True,
               source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (HERE/'greedy_planner.py', HERE/'policy_adapter.py', Path(__file__))})
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'checks_result.json'
    output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
