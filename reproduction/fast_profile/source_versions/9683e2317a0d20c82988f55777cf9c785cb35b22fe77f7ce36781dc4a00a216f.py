"""Paired old/new cold profile timing with identical query-local caches."""
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from dataclasses import asdict
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RELEASE=ROOT/'artifacts/CERTIS_release_20261104'
sys.path[:0]=[str(ROOT/'code'),str(RELEASE/'reproduction/runtime/greedy_count_profile')]
from count_mask_planner import SideCounts, _numpy_profile
from piecewise_count_profile import piecewise_profile
from piecewise_numpy_profile import numpy_piecewise_profile
from greedy_planner import allocate_profiles

def states(spec):
    n,k,kind=spec['n'],8,spec['kind']
    weights=[(i+1)**-1.5 if kind=='skewed' else 1+(i%7)/11 if kind!='repeated' else 1 for i in range(k)]
    totals=[int(n*w/sum(weights)) for w in weights]
    for i in range(n-sum(totals)):totals[i%k]+=1
    snapshots=[]
    for snapshot in range(4 if kind=='refinement' else 1):
        pairs=[]
        for i,total in enumerate(totals):
            nl=total//(32 if kind=='imbalanced' else 2);nr=total-nl
            pair=[]
            for j,original in enumerate((nl,nr)):
                base_h=0 if kind=='repeated' else int(original*(.02+.01*((i+j)%3)))
                lower=base_h+math.floor(original*(.35 if kind=='repeated' else .15+.03*((i+j)%7)))
                upper=base_h+math.ceil(original*(.65 if kind=='repeated' else .65+.02*((i+2*j)%5)))
                removed=int(original*snapshot*.08)
                unknown=original-removed;h=base_h+removed//2
                forced=0 if kind=='repeated' else min(unknown,int(original*.01))
                pair.append(SideCounts(lower,upper,h,unknown,forced))
            pairs.append(tuple(pair))
        snapshots.append(tuple(pairs))
    return snapshots

def worker(spec):
    snapshots=states(spec)
    all_rows=[];expected=None
    for repetition in range(3):
        for backend in (('old','piecewise') if repetition%2==0 else ('piecewise','old')):
            builder=numpy_piecewise_profile if spec.get('variant')=='vectorized' else piecewise_profile
            fn=(lambda a,b:_numpy_profile(a,b,1,10)) if backend=='old' else (
                lambda a,b:builder(a,b,Fraction(1,10)))
            cached=lru_cache(maxsize=65536)(fn)
            profile_seconds=heap_seconds=0.;profiles_all=[];plans=[]
            for keys in snapshots:
                start=time.perf_counter()
                profiles=tuple(cached(a,b) for a,b in keys)
                built=time.perf_counter();plan=allocate_profiles(profiles)
                done=time.perf_counter()
                profile_seconds+=built-start;heap_seconds+=done-built
                profiles_all.append(profiles)
                plans.append(None if plan is None else (plan.discarded,plan.support,plan.counts))
            signature=(profiles_all,plans)
            if expected is None:expected=signature
            else:assert signature==expected,'Profile value, witness or heap decision mismatch'
            all_rows.append(dict(backend=backend,repetition=repetition,profile_seconds=profile_seconds,
                heap_seconds=heap_seconds,total_seconds=profile_seconds+heap_seconds,
                cache=cached.cache_info()._asdict()))
            cached.cache_clear()
    medians={b:statistics.median(r['total_seconds'] for r in all_rows if r['backend']==b)
             for b in ('old','piecewise')}
    record=dict(status='pass',**spec,rows=all_rows,medians=medians,
        speedup=medians['old']/medians['piecewise'],snapshots=len(snapshots),
        unique_states=len({s for snap in snapshots for s in snap}),
        max_side_unknown=max(s.unknown for snap in snapshots for pair in snap for s in pair),
        states=[[[asdict(s) for s in pair] for pair in snap] for snap in snapshots],
        profile_and_allocation_equal=True)
    print(json.dumps(record))

def run(vectorized=False):
    destination=HERE/('benchmark_vectorized' if vectorized else 'benchmark');destination.mkdir(exist_ok=True)
    specs=[dict(n=n,kind=kind) for n in (4096,32768,131072)
           for kind in ('repeated','heterogeneous','skewed','imbalanced')]
    specs += [dict(n=n,kind='refinement') for n in (32768,131072)]
    if vectorized:
        for spec in specs:spec['variant']='vectorized'
    binding={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('code/count_mask_planner.py','code/piecewise_count_profile.py','code/piecewise_numpy_profile.py',
         'research/icde2027_fast_profile_20261106/benchmark_profile.py')}
    manifest=destination/'binding.json'
    if manifest.exists():assert json.loads(manifest.read_text())==binding
    else:manifest.write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
    records=[]
    for spec in specs:
        target=destination/f"n{spec['n']}_{spec['kind']}.json"
        if target.exists():record=json.loads(target.read_text())
        else:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(spec)],
                capture_output=True,text=True,timeout=180)
            if result.returncode:raise RuntimeError(result.stderr[-3000:])
            record=json.loads(result.stdout)
            target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        records.append(record)
        print(json.dumps({key:record[key] for key in ('n','kind','medians','speedup','unique_states')}),flush=True)
    (destination/'summary.json').write_text(json.dumps(dict(status='pass',records=records,binding=binding,
        scope='Synthetic fixed-count snapshots; all three cold trials include identical within-query memoization. '
              'Refinement cases have four sequential snapshots. Timing excludes fixture construction and equality checks.'),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker');parser.add_argument('--vectorized',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(json.loads(args.worker))
    else:run(args.vectorized)
