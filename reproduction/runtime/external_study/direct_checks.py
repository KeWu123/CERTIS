"""Independent Boolean-world and charged-execution checks of direct driver."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_tiny_case
from direct_arms import Envelope, run_direct, downward_alpha, DELTA


def bag(case, truth, discarded=()):
    drop = set(discarded)
    return {(k, a, b) for k, (left, right) in enumerate(zip(case.key_left, case.key_right))
            for a in left for b in right if truth[a] and truth[b] and a not in drop and b not in drop}


def main():
    started = time.perf_counter(); checks = 0; executions = 0
    rng = np.random.default_rng(261016)
    for repeat in range(80):
        n = 6
        multikey = repeat % 2 == 1
        if multikey:
            left = ((0, 1), (1, 2)); right = ((3, 4), (4, 5))
        else:
            left = ((0, 1), (2,)); right = ((3, 4), (5,))
        truth = rng.integers(0, 2, n).tolist()
        case = make_tiny_case(side=[0]*3+[1]*3, key_left=left, key_right=right,
                              truth=truth, name=f"check{repeat}")
        order = rng.permutation(n)
        facts = {j: truth[j] for j in range(n) if rng.random() < .4}
        intervals = {}
        for ids in (*left, *right):
            positive = sum(truth[j] for j in ids)
            intervals[ids] = (int(rng.integers(0, positive+1)),
                              int(rng.integers(positive, len(ids)+1)))
        env = Envelope(case, intervals, order)
        for t in range(n+1):
            drop = set(map(int, order[:t])) - facts.keys()
            maximum = None
            for z in product((0, 1), repeat=n):
                if any(z[j] != y for j, y in facts.items()): continue
                if any(not lo <= sum(z[j] for j in ids) <= hi for ids, (lo, hi) in intervals.items()): continue
                full = bag(case, z); kept = bag(case, z, drop)
                sign = 9*len(full)-10*len(kept)
                maximum = sign if maximum is None else max(maximum, sign)
            assert maximum is not None
            bound = env.support(facts, t)
            assert bound >= maximum
            if not multikey: assert bound == maximum
            checks += 1
    for truth in product((0, 1), repeat=6):
        case = make_tiny_case(side=[0]*3+[1]*3,
            key_left=((0, 1), (1, 2)), key_right=((3, 4), (4, 5)),
            truth=truth, scores=[0., .7, 1., .2, .9, .3], name="execute")
        full = bag(case, truth)
        for audit in (False, True):
            arms = [run_direct(case, 101, 1, audit=audit, feedback=fb,
                               materialize=False) for fb in (False, True)]
            for result in arms:
                returned = bag(case, [result["facts"].get(j, 0) for j in range(6)])
                # Confidence failures, if any, are not asserted away. On the
                # observed joint-coverage event compare actual bag loss exactly.
                covered = all(r["lower"] <= sum(truth[j] for j in r["ids"]) <= r["upper"]
                              for r in result["regions"])
                if covered: assert 10*(len(full)-len(returned)) <= len(full)
                assert returned == bag(case, truth, result["discarded"])
                assert result["summary"]["oracle_requests"] == len(result["oracle_ledger"])
                assert result["summary"]["oracle_requests"] <= 6
                assert sum((Fraction(r["alpha_executed"]) for r in result["regions"]), Fraction(0)) <= DELTA
                executions += 1
            static_paid = {r["id"] for r in arms[0]["oracle_ledger"]}
            feedback_paid = {r["id"] for r in arms[1]["oracle_ledger"]}
            assert feedback_paid <= static_paid
        case.close_sql()
    # Exact budget must fit below a nominal rational even when its nearest
    # IEEE value rounds upward (1/20 is the simplest relevant example).
    for events in range(1, 1000):
        target = DELTA/events
        assert Fraction.from_float(downward_alpha(target)) <= target
    source = Path(__file__).with_name("direct_arms.py")
    out = {"status": "pass", "identity_support_checks": checks,
           "complete_executions": executions, "downward_budget_checks": 999,
           "support_reference": "Boolean assignment and actual derivation enumeration, no polygons",
           "scope": "tiny fixtures; full external matrix not covered by these checks",
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "elapsed_seconds": time.perf_counter()-started}
    Path(__file__).with_name("direct_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__": main()
