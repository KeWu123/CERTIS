"""Separate certificate strength from greedy allocation quality on the full grid."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import count_experiments as experiment
from count_experiments import SideCounts, profiles, study


def optimum(local):
    scores=np.array([0],dtype=np.int64)
    for profile in local:
        values=np.full(len(scores)+len(profile.values)-1,np.iinfo(np.int64).max,dtype=np.int64)
        for extra,cost in enumerate(profile.values):
            target=values[extra:extra+len(scores)]
            np.minimum(target,scores+cost,out=target)
        scores=values
    feasible=np.flatnonzero(scores<=0)
    return None if not len(feasible) else int(feasible[-1])+sum(p.forced for p in local)


def run(out):
    source=out/'mechanism.json'
    rows=json.loads(source.read_text(encoding='utf-8'))['rows']
    result=[]
    for index,row in enumerate(rows):
        keys=tuple(tuple(SideCounts(**s) for s in pair) for pair in row['count_states'])
        joint, separate=profiles(keys),profiles(keys,True)
        jd,sd=optimum(joint),optimum(separate)
        assert jd>=sd
        assert jd>=row['joint_discard'] and sd>=row['separate_discard']
        if row['n']==128:
            reference=experiment.exact.plan_counts(keys,experiment.Fraction(1,10))
            assert jd==reference.discarded
        result.append(dict(**row,joint_optimal_discard=jd,separate_optimal_discard=sd,
            exact_certificate_gain=jd-sd))
        if index%20==0:print(json.dumps(dict(completed=index+1,expected=len(rows))),flush=True)
    study.save(out/'mechanism_optima.json',dict(status='pass',rows=result,cells=len(rows),
        source_sha256=study.sha(source),driver_sha256=study.sha(__file__),
        reference_check='Joint profile convolution checked against production DP for all N=128 cells.',
        scope='Exact fixed-cache optima separate certificate strength from the same heap heuristic.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=experiment.ROOT/'outputs/count_completion_20261106')
    run(p.parse_args().output)
