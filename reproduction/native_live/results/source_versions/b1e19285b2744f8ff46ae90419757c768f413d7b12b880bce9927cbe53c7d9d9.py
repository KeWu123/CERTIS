"""Definition-only checks: Boolean completions and exhaustive count pairs."""
from itertools import product
from fractions import Fraction
import json
from pathlib import Path

from adaptive_marginal import Side, recall_pair, continuous_plan, satisfies


def brute_recall(s, d):
    low, high = s.bounds()
    values = []
    for labels in product((0, 1), repeat=s.unknown):
        total = s.positive + sum(labels)
        if low <= total <= high:
            values.append(Fraction(s.positive + sum(labels[d:]), total) if total else Fraction(1))
    return min(values)


def run():
    sides = []
    count = 0
    for n in range(5):
        for h in range(3):
            for low in range(h+n+1):
                for high in range(low, h+n+1):
                    s = Side(low, high, h, n)
                    try: s.bounds()
                    except ValueError: continue
                    for d in range(n+1):
                        assert Fraction(*recall_pair(s, d)) == brute_recall(s, d)
                        count += 1
                    sides.append(s)
    plans = 0
    # Independent full pair enumeration, including forced-discard constraints.
    for a in sides[::3]:
        for b in sides[::5]:
            for fa in (0, a.unknown):
                for fb in (0, b.unknown):
                    x = Side(a.lower,a.upper,a.positive,a.unknown,fa)
                    y = Side(b.lower,b.upper,b.positive,b.unknown,fb)
                    valid = [(ld, rd) for ld in range(fa,a.unknown+1)
                             for rd in range(fb,b.unknown+1)
                             if brute_recall(x,ld)*brute_recall(y,rd) >= Fraction(9,10)]
                    expected = max(valid,key=lambda p:(sum(p),-p[0])) if valid else None
                    assert continuous_plan(x,y) == expected, (x,y,expected,continuous_plan(x,y))
                    plans += 1
    report = dict(status='pass',boolean_side_checks=count,exhaustive_pair_checks=plans)
    Path(__file__).with_name('marginal_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__': run()
