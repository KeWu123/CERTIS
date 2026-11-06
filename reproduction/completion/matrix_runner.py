"""Fresh complete-result runners for the paper's matched and structural studies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/study'),
    str(ROOT/'reproduction/runtime/external_study'),
    str(ROOT/'reproduction/runtime/greedy_count_profile'),
    str(ROOT/'reproduction/runtime/structural_study'),str(ROOT/'code')]
import run_study as study
import count_mask_planner as exact
from count_mask_planner import CountProfile
from greedy_planner import allocate_profiles
from bilinear_certificate import InconsistentEvidence

MATCHED=('fixed_grid','adaptive_grid','fixed_continuous','adaptive_continuous',
         'separate_greedy','joint_greedy','joint_dp')


def allocation(case,seed,budget,separate):
    spec=importlib.util.spec_from_file_location('_fresh_count_execution',study.EXTERNAL/'count_allocation_arms.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    instances=[]
    class Tracked(study.BatchOracle):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);instances.append(self)
    module.Oracle=Tracked
    observations=[]
    if separate:
        def plan(keys,allowance):
            p,q=exact._fraction(allowance)
            local=[]
            for a,b in keys:
                loss=exact._numpy_profile(a,b,0,1)
                lower=max(a.lower,a.positive)*max(b.lower,b.positive)
                local.append(CountProfile(loss.forced,tuple(q*v-p*lower for v in loss.values),loss.choices))
            answer=allocate_profiles(local)
            if answer is None:return None
            joint_support=0
            for (a,b),(dl,dr) in zip(keys,answer.counts):
                left=exact.RegionSide(a.lower,a.upper,a.positive,dl,a.unknown-dl)
                right=exact.RegionSide(b.lower,b.upper,b.positive,dr,b.unknown-dr)
                joint_support+=exact.frontier_support(left,right,allowance)
            assert joint_support<=answer.support<=0
            observations.append(dict(separate_support=answer.support,joint_support=joint_support,
                discarded=answer.discarded))
            # The executor independently checks the joint support of the
            # selected mask. Selection itself used only separate extrema.
            return replace(answer,support=joint_support)
        module.plan_counts=plan
    result=module.run_count_allocation(case,seed,budget,materialize=True)
    result['batches']=instances[0].batches
    result['method']='separate_greedy' if separate else 'joint_dp'
    if separate:result['separate_support_observations']=observations
    return result


def matched(args):
    cases=study.cases_for('constructed')
    if args.pilot:cases=cases[:1]
    selected=args.methods.split(',') if args.methods else MATCHED
    assert set(selected)<=set(MATCHED)
    rows=[]
    expected=json.loads((ROOT/'reproduction/archive/allowance/sep_extrema_results.json').read_text())
    expected={(r['case_id'],r['seed']):r['requests'] for r in expected}
    source=study.sha(__file__)
    for case in cases:
        for seed in ((101,) if args.pilot else (101,202,303)):
            for method in selected:
                path=args.output/'raw'/f'{case.case_id}__{method}__s{seed}__b500.json.gz'
                if path.exists():
                    record=study.read(path);assert record['driver_sha256']==source
                    result=record['result']
                else:
                    started=time.perf_counter()
                    if method in MATCHED[:4]:result=study.marginal(case,seed,500,method)
                    elif method=='joint_greedy':result=study.greedy(case,seed,500)
                    else:result=allocation(case,seed,500,method=='separate_greedy')
                    record=dict(status='pass',case_id=case.case_id,method=method,seed=seed,budget=500,
                        driver_sha256=source,target_version=case.target_version,oracle_version=case.oracle_version,
                        result=result,wall_seconds=time.perf_counter()-started)
                    study.save(path,record,True)
                assert result['output']['bag_subset']
                if method=='separate_greedy':assert result['summary']['oracle_requests']==expected[case.case_id,seed]
                row=dict(case_id=case.case_id,method=method,seed=seed,
                    requests=result['summary']['oracle_requests'],loss=result['output']['loss'],
                    full_count=result['output']['full_count'],returned_count=result['output']['returned_count'],
                    raw=path.relative_to(args.output).as_posix(),sha256=study.sha(path))
                rows.append(row)
                print(json.dumps(dict(completed=len(rows),**row)),flush=True)
        case.close_sql()
    study.save(args.output/'summary.json',dict(status='pass',rows=rows,cells=len(rows),methods=selected))


def structural(args):
    import run_structural as original
    import structural_cases as generation
    specs=generation.full_grid()
    if args.pilot:specs=specs[:1]
    rows=[]
    for spec in specs:
        case=generation.build(spec)
        methods=original.METHODS if not spec['reuse'] else original.METHODS[:-1]
        for method in methods:
            path=args.output/'raw'/(spec['target']+'__'+method+'.json.gz')
            if path.exists():
                record=study.read(path);assert record['driver_sha256']==study.sha(__file__)
            else:
                case.close_sql()
                result=original.run_one(case,method,spec['seed'],max(1,spec['n']//16))
                validation=original.validate(spec,case,result,method,explicit=args.pilot)
                record=dict(driver_sha256=study.sha(__file__),input_sha256=spec['input_sha256'],
                    target=spec['target'],method=method,result=result,validation=validation)
                study.save(path,record,True)
            result=record['result'];summary=result.get('summary',result)
            rows.append(dict(target=spec['target'],method=method,requests=summary['oracle_requests'],
                loss=result['output']['loss'],validation=record['validation']))
            print(json.dumps(dict(completed=len(rows),target=spec['target'],method=method)),flush=True)
        case.close_sql()
    study.save(args.output/'summary.json',dict(status='pass',rows=rows,cells=len(rows)))


def legacy(args):
    from count_allocation_arms import eligible, run_count_allocation
    from direct_arms import run_direct
    from cut_arms import run_cut
    from exact_baselines import run_exact, METHODS as EXACT
    from extra_exact_controls import run_extra_exact
    from mechanism_controls import run_mechanism, METHODS as MECHANISMS
    if args.kind=='native_original':
        from native_discussion_cases import load_native_cases
        cases=load_native_cases()
    else:
        cases=study.common.load_cases()
        if args.kind=='primary_count':cases=[c for c in cases if eligible(c)]
    if args.pilot:cases=cases[:1]
    rows=[]
    for case in cases:
        jobs=[]
        seeds=(101,) if args.pilot else (101,202,303)
        if args.kind=='mechanisms':
            jobs=[(m,s,500,lambda m=m,s=s:run_mechanism(case,m,s,500)) for m in MECHANISMS for s in seeds]
        elif args.kind=='extra':
            jobs=[(m,0,0,lambda m=m:run_extra_exact(case,m))
                  for m in ('dynamic_max_degree','helper_negative_first')]
        elif args.kind=='primary_count':
            jobs=[('count_audit',s,b,lambda s=s,b=b:run_count_allocation(case,s,b))
                  for b in (125,500) for s in seeds]
            jobs.append(('count_point',101,0,lambda:run_count_allocation(case,101,0,audit=False)))
        else:
            jobs=[(m,0,0,lambda m=m:run_exact(case,m)) for m in EXACT]
            jobs += [(m,0,0,lambda m=m:run_extra_exact(case,m))
                     for m in ('dynamic_max_degree','helper_negative_first')]
            jobs += [('degree_point',101,0,lambda:run_direct(case,101,0,audit=False,helper=False)),
                     ('count_point',101,0,lambda:run_count_allocation(case,101,0,audit=False))]
            for b in (125,500):
                for s in seeds:
                    jobs += [('degree_audit',s,b,lambda s=s,b=b:run_direct(case,s,b,helper=False)),
                             ('count_audit',s,b,lambda s=s,b=b:run_count_allocation(case,s,b)),
                             ('cut_audit',s,b,lambda s=s,b=b:run_cut(case,s,b))]
        for method,seed,budget,work in jobs:
            path=args.output/'raw'/f'{case.case_id}__{method}__s{seed}__b{budget}.json.gz'
            if path.exists():
                record=study.read(path);assert record['driver_sha256']==study.sha(__file__)
            else:
                result=work()
                record=dict(driver_sha256=study.sha(__file__),case_id=case.case_id,method=method,
                    seed=seed,budget=budget,result=result,status='pass')
                study.save(path,record,True)
            result=record['result'];summary=result.get('summary',result)
            assert result['output']['bag_subset']
            rows.append(dict(case_id=case.case_id,method=method,seed=seed,budget=budget,
                requests=summary['oracle_requests'],loss=result['output']['loss']))
            print(json.dumps(dict(completed=len(rows),**rows[-1])),flush=True)
        case.close_sql()
    study.save(args.output/'summary.json',dict(status='pass',cells=len(rows),rows=rows))


def geometry(args):
    import run_structural as original
    source=ROOT/'reproduction/archive/structural'
    original.HERE=args.output
    for name in ('geometry_inputs.json',):
        (args.output/name).write_bytes((source/name).read_bytes())
    binding=dict(driver_sha256=study.sha(__file__),
        original_driver_sha256=study.sha(original.__file__),
        inputs_sha256=study.sha(args.output/'geometry_inputs.json'))
    study.save(args.output/'binding.json',binding)
    def verify_current():
        assert binding['driver_sha256']==study.sha(__file__)
        assert binding['original_driver_sha256']==study.sha(original.__file__)
        assert binding['inputs_sha256']==study.sha(args.output/'geometry_inputs.json')
        return binding
    original.verify_freeze=verify_current
    original.geometry()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['matched','structural','primary_count','mechanisms','extra','native_original','geometry'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods')
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__).resolve()]
    for directory in (ROOT/'code',ROOT/'reproduction/study',
        ROOT/'reproduction/runtime/external_study',ROOT/'reproduction/runtime/greedy_count_profile',
        ROOT/'reproduction/runtime/structural_study'):
        files.extend(directory.glob('*.py'))
    binding=dict(sources={p.relative_to(ROOT).as_posix():study.sha(p) for p in files},
        input_manifest_sha256=study.sha(study.common.SOURCE/'download_manifest.json'),
        native_mapping_sha256=study.sha(ROOT/'reproduction/runtime/native_discussion/relation_metadata.parquet'),
        kind=args.kind,methods=args.methods,pilot=args.pilot)
    frozen=args.output/'binding.json'
    if frozen.exists():assert json.loads(frozen.read_text(encoding='utf-8'))==binding
    else:study.save(frozen,binding)
    if args.kind in ('matched','structural','geometry'):globals()[args.kind](args)
    else:legacy(args)
