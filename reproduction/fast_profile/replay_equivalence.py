"""Check old/new profiles and execution ledgers on all native DBpedia queries.

These executions include additional validation overhead and are not latency
benchmarks. The archival quality and live-inference matrices are unchanged.
"""
from pathlib import Path
from fractions import Fraction
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RELEASE=ROOT if (ROOT/'reproduction/completion').is_dir() else ROOT/'artifacts/CERTIS_release_20261104'
sys.path[:0]=[str(RELEASE/'reproduction/study'),str(RELEASE/'reproduction/completion'),str(ROOT/'code')]
import native_dbpedia as native
import cached_greedy
from piecewise_numpy_profile import numpy_piecewise_profile

def digest(value):
    normalized=json.loads(json.dumps(value,default=native.study.convert))
    return hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()

def run(destination):
    os.chdir(RELEASE)
    destination.mkdir(parents=True,exist_ok=True)
    cases=native.cases_for()
    binding=dict(protocol='All 16 native DBpedia queries; audit 125/500; seed 101; matched old/new cached greedy.',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_arrays_sha256=native.study.sha(native.common.INPUTS/'dbpedia_arrays.npz'),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('code/piecewise_count_profile.py','code/piecewise_numpy_profile.py','code/count_mask_planner.py')},
        targets={c.case_id:c.target_version for c in cases},seed=101,budgets=[125,500])
    manifest=destination/'binding.json'
    if manifest.exists():assert json.loads(manifest.read_text())==binding
    else:manifest.write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
    original=cached_greedy.exact._numpy_profile
    rows=[];states={}
    for case in cases:
        for budget in (125,500):
            key=f'{case.case_id}__s101__b{budget}'
            path=destination/'raw'/(key+'.json.gz')
            if path.exists():record=native.study.read(path)
            else:
                case.close_sql();old=cached_greedy.execute(case,101,budget)
                checks=[]
                def checked(a,b,p,q):
                    observed=numpy_piecewise_profile(a,b,Fraction(p,q))
                    expected=original(a,b,p,q)
                    assert observed==expected,'Profile values or lexicographic witnesses differ'
                    state=dict(left=asdict(a),right=asdict(b),p=p,q=q)
                    state_id=digest(state);states[state_id]=state;checks.append(state_id)
                    return observed
                case.close_sql();cached_greedy.exact._numpy_profile=checked
                try:new=cached_greedy.execute(case,101,budget)
                finally:cached_greedy.exact._numpy_profile=original
                fields=('oracle_ledger','facts','discarded','events','regions','batches')
                signatures={name:digest(old[name]) for name in fields}
                for name in fields:
                    # Event planner timings are intentionally separate from
                    # semantic event contents; current events contain counts.
                    assert digest(new[name])==signatures[name],(key,name)
                for name in ('full_count','returned_count','lost_count','bag_subset'):
                    assert old['output'][name]==new['output'][name],(key,name)
                reference=RELEASE/'reproduction/completion/results/native_dbpedia/raw'/(
                    f'{case.case_id}__joint_greedy_cached__s101__b{budget}.json.gz')
                archived=native.study.read(reference)['result']
                for name in ('oracle_ledger','facts','discarded','regions','batches'):
                    assert digest(archived[name])==signatures[name],(key,'archive',name)
                record=dict(status='pass',case_id=case.case_id,budget=budget,seed=101,
                    requests=len(new['oracle_ledger']),profile_checks=len(checks),states=checks,
                    signatures=signatures,full_count=new['output']['full_count'],
                    returned_count=new['output']['returned_count'],
                    archive_path=reference.relative_to(RELEASE).as_posix(),
                    archive_sha256=native.study.sha(reference),
                    semantics_and_order_equal=True,latency_scope='Validation overhead included; not a timing benchmark')
                native.study.save(path,record,True)
                native.study.save(destination/'states.json',states)
            rows.append(record)
            print(json.dumps(dict(completed=len(rows),case_id=case.case_id,budget=budget,
                requests=record['requests'],profile_checks=record['profile_checks'])),flush=True)
        case.close_sql()
    if (destination/'states.json').exists():states=json.loads((destination/'states.json').read_text())
    native.study.save(destination/'summary.json',dict(status='pass',matched_cells=len(rows),
        executions=2*len(rows),unique_profile_states=len(states),rows=rows,binding=binding))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/fast_profile_equivalence')
    args=parser.parse_args();run(args.output.resolve())
