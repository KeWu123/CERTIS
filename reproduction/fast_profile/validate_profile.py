"""Independent exhaustive checks for the opt-in exact profile prototype."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import random
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from count_mask_planner import SideCounts, _python_profile, _numpy_profile, _int64_safe
from piecewise_count_profile import piecewise_profile, _piece_minimum
from piecewise_numpy_profile import numpy_piecewise_profile
sys.path.insert(0,str(ROOT/'reproduction/runtime/greedy_count_profile'))
from greedy_planner import allocate_profiles
from count_mask_planner import plan_counts

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def side(rng, maximum=12, big=False):
    unknown=rng.randrange(maximum+1)
    positive=rng.randrange(5)
    if big:positive*=10**40
    total=positive+rng.randrange(unknown+1)
    lower=rng.randrange(total+1) if total else 0
    upper=total+rng.randrange(maximum+2)
    return SideCounts(lower,upper,positive,unknown,rng.randrange(unknown+1))

def main():
    start=time.perf_counter();rng=random.Random(20261106)
    small=[]
    for n in range(2):
        for h in range(2):
            for lower in range(h+n+1):
                for upper in range(max(lower,h),h+n+2):
                    for forced in range(n+1):small.append(SideCounts(lower,upper,h,n,forced))
    cases=[(a,b,Fraction(p,10)) for a in small for b in small for p in (0,1,5,10)]
    for index in range(5000):
        q=rng.choice((1,10,13,97,10**60+7))
        cases.append((side(rng,big=index%17==0),side(rng,big=index%19==0),Fraction(rng.randrange(q+1),q)))
    for index,(a,b,allowance) in enumerate(cases):
        observed=piecewise_profile(a,b,allowance)
        expected=_python_profile(a,b,allowance)
        assert observed==expected,(index,a,b,allowance,observed,expected)
        p,q=allowance.numerator,allowance.denominator
        if _int64_safe(((a,b),),p,q):assert observed==_numpy_profile(a,b,p,q),index
        if index<1000 or 4096<=index<5096:
            assert observed==numpy_piecewise_profile(a,b,allowance,small_left_cutoff=-1),index
        if index and index%2000==0:print(json.dumps(dict(profiles_checked=index)),flush=True)
    profile_seconds=time.perf_counter()-start
    for index in range(25000):
        scale=10**80 if index%7==0 else 1
        polynomials=[tuple(rng.randint(-100,100)*scale for _ in range(3))
                     for _ in range(rng.randint(1,4))]
        lower=rng.randint(-50,30);upper=lower+rng.randint(0,60)
        observed=_piece_minimum(polynomials,lower,upper)
        expected=min((max((a*x+b)*x+c for c,b,a in polynomials),x)
                     for x in range(lower,upper+1))
        assert observed==expected,(index,polynomials,lower,upper,observed,expected)
    report=dict(status='pass',profile_instances=len(cases),exhaustive_small_instances=len(cases)-5000,
        random_profile_instances=5000,independent_envelopes=25000,
        exact_values_and_lexicographic_witnesses_match=True,
        forced_vectorized_profile_instances=2000,
        profile_seconds=profile_seconds,total_seconds=time.perf_counter()-start,
        source_sha256=sha(ROOT/'code/piecewise_count_profile.py'),
        vectorized_source_sha256=sha(ROOT/'code/piecewise_numpy_profile.py'),
        reference_sha256=sha(ROOT/'code/count_mask_planner.py'),seed=20261106)
    examples=[]
    for k in (2,3,10,32):
        left=SideCounts(1,1,0,1)
        keys=[(left,SideCounts(0,2,1,1)),(left,SideCounts(1,1,0,k-1))]
        keys += [(SideCounts(1,1,1,0),SideCounts(1,1,1,0))]*8
        profiles=tuple(piecewise_profile(a,b,Fraction(1,10)) for a,b in keys)
        greedy=allocate_profiles(profiles);optimum=plan_counts(keys,Fraction(1,10))
        assert profiles[0].values==(-1,8,18) and profiles[1].values==(-1,)+(9,)*k
        assert greedy.discarded==1 and optimum.discarded==k
        examples.append(dict(k=k,greedy=greedy.discarded,optimum=optimum.discarded))
    report['recall_90_counterexamples']=examples
    report['validation_driver_sha256']=sha(Path(__file__))
    output=ROOT/'outputs/fast_profile_validation.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
