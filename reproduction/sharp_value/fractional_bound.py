"""Exact fractional multiple-choice relaxation of fixed-cache count profiles."""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import random

from sharp_wrapper import HERE, build_profile, exact, study


def fractional_upper_bound(profiles):
    """Upper-bound feasible integer cardinality, including forced discards."""
    profiles=tuple(profiles)
    budget=-sum(p.values[0] for p in profiles)
    if budget<0:
        return None
    edges=[]
    forced=sum(p.forced for p in profiles)
    for index,profile in enumerate(profiles):
        assert all(a<=b for a,b in zip(profile.values,profile.values[1:]))
        base=profile.values[0]
        hull=[]
        for g,value in enumerate(profile.values):
            point=(g,value-base)
            while len(hull)>=2:
                ga,ca=hull[-2];gb,cb=hull[-1];gc,cc=point
                if (cb-ca)*(gc-gb)>=(cc-cb)*(gb-ga):
                    hull.pop()
                else:
                    break
            hull.append(point)
        for order,((ga,ca),(gb,cb)) in enumerate(zip(hull,hull[1:])):
            edges.append((Fraction(cb-ca,gb-ga),index,order,gb-ga,cb-ca))
    extra=Fraction(0)
    remaining=budget
    for slope,index,order,gain,cost in sorted(edges):
        if cost<=remaining:
            extra+=gain;remaining-=cost
        else:
            extra+=Fraction(remaining*gain,cost)
            break
    return Fraction(forced)+extra


def validate():
    rng=random.Random(20261107)
    cases=2000
    for trial in range(cases):
        profiles=[]
        for _ in range(rng.randint(1,4)):
            values=[-rng.randint(0,12)]
            for g in range(rng.randint(0,5)):
                values.append(values[-1]+rng.randint(0,12))
            profiles.append(exact.CountProfile(rng.randint(0,3),tuple(values),tuple()))
        bound=fractional_upper_bound(profiles)
        optimum=max(sum(p.forced+g for p,g in zip(profiles,choice))
            for choice in product(*(range(len(p.values)) for p in profiles))
            if sum(p.values[g] for p,g in zip(profiles,choice))<=0)
        assert optimum<=bound.numerator//bound.denominator
        assert bound<=sum(p.forced+len(p.values)-1 for p in profiles)
    return dict(status='pass',cases=cases,seed=20261107,
        check='fractional upper bound dominates brute-force integer optimum')


def main(args):
    validation=validate()
    out=args.results.resolve()
    summary=json.loads((out/'summary.json').read_text())
    rows=[]
    for row in summary['rows']:
        path=out/row['raw']['joint']['path']
        assert study.sha(path)==row['raw']['joint']['sha256']
        snapshot=study.read(path)['result']['sharp_snapshots'][0]
        keys=tuple((exact.SideCounts(**k['left']),exact.SideCounts(**k['right']))
            for k in snapshot['keys'])
        profiles=tuple(build_profile(a,b,Fraction(1,10),True) for a,b in keys)
        bound=fractional_upper_bound(profiles)
        assert bound is not None
        upper=bound.numerator//bound.denominator
        joint=snapshot['alternatives']['joint']['discarded']
        separate=snapshot['alternatives']['separate']['discarded']
        # All native rows have category membership: no orphan additions. The
        # initial evidence has no forced old discards either. Keep the count
        # convention explicit before comparing to the profile-only relaxation.
        assert joint==row['initial_joint_discard']
        assert separate==row['initial_separate_discard']
        assert all(s.forced_discard==0 for pair in keys for s in pair)
        assert separate<=upper
        rows.append(dict(case_id=row['case_id'],key=row['key'],budget=row['budget'],seed=101,
            joint_greedy_discard=joint,separate_greedy_discard=separate,
            separate_fractional_upper_bound=str(bound),separate_integer_upper_bound=upper,
            proven_cardinality_advantage=max(0,joint-upper),
            raw_joint_sha256=row['raw']['joint']['sha256']))
    result=dict(status='pass',secondary_post_hoc_diagnostic=True,validation=validation,
        cells=len(rows),rows=rows,
        cells_with_proven_cardinality_advantage=sum(r['proven_cardinality_advantage']>0 for r in rows),
        max_proven_cardinality_advantage=max(r['proven_cardinality_advantage'] for r in rows),
        interpretation='The fractional result is an upper bound, not an executable policy. '
            'These are initial fixed-cache discards, not adaptive request optima.')
    study.save(out/'fractional_bound.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=HERE/'results')
    main(p.parse_args())
