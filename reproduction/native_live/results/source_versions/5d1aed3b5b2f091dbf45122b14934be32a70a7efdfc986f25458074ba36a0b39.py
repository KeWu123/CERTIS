"""Resumable, hash-bound split/cost experiments; historical sources read only."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTERNAL = ROOT/'reproduction/runtime/external_study'
GREEDY = ROOT/'reproduction/runtime/greedy_count_profile'
sys.path[:0] = [str(HERE),str(EXTERNAL),str(GREEDY),str(ROOT/'code')]
from adaptive_marginal import Side, recall_pair, continuous_plan, cap_at_recall, grid_plan
import common
from direct_arms import make_regions
from count_allocation_arms import CountEnvelope, eligible
from policy_adapter_v3 import run_greedy_count
import policy_adapter_v2
from exact_baselines import LivePartners
from strict_result_validation import load_pinned_reference
from strict_result_validation_v2 import strict_check_record
from bilinear_certificate import InconsistentEvidence

PAIRS = (('sent_positive','sent_negative'),('approval','disapproval'),('joy','sadness'),
         ('optimism','fear'),('love','disgust'),('gratitude','disappointment'))
SEEDS = (101,202,303)
SPLIT_METHODS = ('fixed_grid','adaptive_grid','fixed_continuous','adaptive_continuous')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(gzip.decompress(Path(p).read_bytes()))
def convert(x):
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)
def save(p,x,compressed=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(x,default=convert,ensure_ascii=False,indent=None if compressed else 2)+'\n').encode()
    if compressed: data=gzip.compress(data,compresslevel=4,mtime=0)
    temp=p.with_suffix(p.suffix+'.tmp'); temp.write_bytes(data); temp.replace(p)


class BatchOracle(common.Oracle):
    """Network batch boundaries; labels still come only from the pinned Oracle."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.batches=[]
    def acquire(self,ids,phase):
        start=len(self.ledger)
        result=super().acquire(ids,phase)
        rows=self.ledger[start:]
        for i in range(0,len(rows),128):
            chunk=rows[i:i+128]
            self.batches.append(dict(phase=phase,ids=[r['id'] for r in chunk],
                size=len(chunk),tokens_in=sum(r['tokens_in'] for r in chunk),
                tokens_out=sum(r['tokens_out'] for r in chunk),
                max_tokens_in=max(r['tokens_in'] for r in chunk),
                max_tokens_out=max(r['tokens_out'] for r in chunk)))
        return result


class MarginalEnvelope:
    def __init__(self,case,intervals,mode):
        if not eligible(case): raise ValueError('Key-disjoint target required')
        self.case,self.intervals,self.mode=case,intervals,mode
        self.incident={j for pair in zip(case.key_left,case.key_right) for ids in pair for j in ids}
        self.fixed={}; self.calls=0; self.changed=0
        self.joint=CountEnvelope(case,intervals)

    def plan(self,facts,old):
        self.calls+=1
        old=set(old)-facts.keys()
        mask=(set(range(self.case.n))-self.incident-facts.keys())|old
        details=[]
        for k,pair in enumerate(zip(self.case.key_left,self.case.key_right)):
            sides=[]; groups=[]
            for ids in pair:
                unknown=[j for j in ids if j not in facts]
                lo,hi=self.intervals.get(ids,(0,0))
                side=Side(lo,hi,sum(facts.get(j)==1 for j in ids),len(unknown),len(old.intersection(unknown)))
                try: side.bounds()
                except ValueError as exc: raise InconsistentEvidence(str(exc)) from exc
                sides.append(side); groups.append(unknown)
            if self.mode.endswith('grid'):
                counts,split=grid_plan(*sides,fixed=self.fixed.get(k) if self.mode.startswith('fixed') else None)
                if k not in self.fixed and split is not None: self.fixed[k]=split
            elif self.mode=='adaptive_continuous' or k not in self.fixed:
                counts=continuous_plan(*sides)
                split=None
                if counts is not None and k not in self.fixed:
                    required=tuple(Fraction(*recall_pair(s,d)) for s,d in zip(sides,counts))
                    self.fixed[k]=((required[0],Fraction(9,10)/required[0])
                                   if self.mode=='fixed_continuous' else required)
            else:
                counts=tuple(cap_at_recall(s,r) for s,r in zip(sides,self.fixed[k]))
                if any(d<s.forced for s,d in zip(sides,counts)): counts=None
                split=None
            if counts is None: raise InconsistentEvidence('No feasible containing per-side plan')
            for unknown,d in zip(groups,counts):
                forced=old.intersection(unknown)
                mask.update(forced)
                mask.update([j for j in unknown if j not in forced][:d-len(forced)])
            details.append(dict(counts=counts,recalls=[str(Fraction(*recall_pair(s,d))) for s,d in zip(sides,counts)]))
        support=self.joint.support(facts,mask)
        if support>0 or not old<=mask: raise AssertionError('Unsafe or noncontaining per-side mask')
        return mask,support,details


def finish(case,oracle,method,seed,budget,mask,initial,regions,timings,**extra):
    t=time.perf_counter(); output=common.evaluate(case,oracle.facts)
    validation_seconds=time.perf_counter()-t
    if common.possible_unknowns(case,oracle.facts,np.array([j not in mask for j in range(case.n)])):
        raise AssertionError('Incomplete retained bag')
    execution={r['id'] for r in oracle.ledger if r['phase']=='execution'}
    if not execution<=initial: raise AssertionError('Request containment failed')
    return dict(method=method,seed=seed,audit_budget=budget,allowance='1/10',delta='1/20',
        summary=oracle.summary(),oracle_ledger=oracle.ledger,helper_ledger=oracle.helper_ledger,
        facts=oracle.facts,regions=regions,discarded=sorted(mask),initial_retained_unknown=sorted(initial),
        request_containment_checked=True,output=output,batches=oracle.batches,timings=timings,
        sql_materialization_and_reference_validation_seconds=validation_seconds,**extra)


def marginal(case,seed,budget,mode):
    oracle=BatchOracle(case); start=time.perf_counter(); prep=start
    rank=common.potential_degrees(case,{})
    order=np.lexsort((np.arange(case.n),-rank))
    preprocessing=time.perf_counter()-prep
    intervals,regions,audit_seconds=make_regions(case,oracle,seed,budget)
    env=MarginalEnvelope(case,intervals,mode)
    t=time.perf_counter(); mask,support,details=env.plan(oracle.facts,set())
    planning=time.perf_counter()-t
    initial=set(range(case.n))-mask-oracle.facts.keys(); conflict=False; bookkeeping=0.
    events=[dict(paid=len(oracle.ledger),counts=details,support=support)]
    while True:
        t=time.perf_counter()
        retained=np.array([j not in mask for j in range(case.n)])
        possible=set(common.possible_unknowns(case,oracle.facts,retained))
        ids=[int(j) for j in order if int(j) in possible][:128]
        bookkeeping+=time.perf_counter()-t
        if not ids: break
        oracle.acquire(ids,'execution')
        if not conflict:
            t=time.perf_counter()
            try: mask,support,details=env.plan(oracle.facts,mask)
            except InconsistentEvidence: conflict=True; support=None
            planning+=time.perf_counter()-t
        events.append(dict(paid=len(oracle.ledger),counts=details,support=support,conflict=conflict))
    try: final_support=env.joint.support(oracle.facts,mask)
    except InconsistentEvidence: final_support=None
    return finish(case,oracle,mode,seed,budget,mask,initial,regions,
        dict(preprocessing_seconds=preprocessing,audit_seconds=audit_seconds,projection_seconds=planning,
             execution_bookkeeping_seconds=bookkeeping,algorithm_seconds=time.perf_counter()-start),
        conflict_freeze=conflict,final_scaled_support=final_support,events=events,projection_calls=env.calls)


def batched_exact(case,method):
    oracle=BatchOracle(case); start=time.perf_counter(); live=LivePartners(case,{})
    prep=time.perf_counter()-start; first=0 if method=='batched_left' else 1
    for side in (first,1-first):
        while True:
            selected=[j for j in range(case.n) if case.side[j]==side and live.eligible(j)][:128]
            if not selected: break
            live.observe(oracle.acquire(selected,'execution'))
    result=finish(case,oracle,method,0,0,set(),set(range(case.n)),[],
        dict(preprocessing_seconds=prep,algorithm_seconds=time.perf_counter()-start),
        no_audit=True,final_scaled_support=0,conflict_freeze=False)
    if not result['output']['exact']: raise AssertionError('Exact batch returned an incomplete answer')
    return result


def greedy(case,seed,budget):
    instances=[]
    class Tracked(BatchOracle):
        def __init__(self,*a,**kw): super().__init__(*a,**kw); instances.append(self)
    module=policy_adapter_v2._execution
    previous=module.Oracle; module.Oracle=Tracked
    try: result=run_greedy_count(case,seed,budget,materialize=True)
    finally: module.Oracle=previous
    if len(instances)!=1: raise AssertionError('Unexpected greedy cache count')
    result['method']='joint_greedy'; result['batches']=instances[0].batches
    return result


def cases_for(kind):
    if kind=='constructed': return [c for c in common.load_cases() if eligible(c)]
    spec=importlib.util.spec_from_file_location('_strengthening_native_cases',EXTERNAL/'native_discussion_cases.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.PAIRS=PAIRS
    return module.load_native_cases()


def source_paths():
    return [HERE/x for x in ('run_study.py','adaptive_marginal.py','PROTOCOL.md','check_marginal.py')]+[
        EXTERNAL/x for x in ('common.py','direct_arms.py','count_allocation_arms.py','native_discussion_cases.py',
        'exact_baselines.py','strict_result_validation.py','strict_result_validation_v2.py','validate_results.py')]+[
        GREEDY/x for x in ('greedy_planner.py','policy_adapter.py','policy_adapter_v2.py','policy_adapter_v3.py')]+[
        ROOT/'code'/x for x in ('bilinear_certificate.py','count_mask_planner.py','exact_hypergeom.py')]


def freeze(out,cases,cells):
    inputs=[common.INPUTS/x for x in ('validation.json','dbpedia_arrays.npz','goemotions_arrays.npz')]
    inputs += [EXTERNAL.parent/'native_discussion'/x for x in ('restoration_manifest.json','relation_metadata.parquet')]
    inputs += [common.SOURCE/'download_manifest.json']
    for item in json.loads(inputs[-1].read_text(encoding='utf-8'))['files']:
        p=common.SOURCE/item['path']; assert sha(p)==item['sha256']; inputs.append(p)
    binding=dict(sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},cells=cells,
        targets={c.case_id:c.target_version for c in cases},preregistered=False,earlier_outcomes_known=True,
        python=sys.version,platform=platform.platform())
    sid=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    manifest=dict(source_id=sid,**binding); path=out/'manifest.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==manifest,'Mixed-source resume forbidden'
    else:
        save(path,manifest)
        for p in source_paths():
            q=out/'sources'/p.relative_to(ROOT); q.parent.mkdir(parents=True,exist_ok=True); q.write_bytes(p.read_bytes())
    return sid


def run(args):
    cases=cases_for(args.kind)
    if args.pilot: cases=cases[:1]
    methods=SPLIT_METHODS if args.kind=='constructed' else ('batched_left','batched_right','adaptive_continuous','joint_greedy')
    cells=[dict(case_id=c.case_id,method=m,seed=s,budget=b) for c in cases for m in methods
           for b in ((0,) if m.startswith('batched') else (500,) if args.kind=='constructed' else (125,500))
           for s in ((0,) if m.startswith('batched') else (101,) if args.pilot else SEEDS)]
    out=HERE/args.output; sid=freeze(out,cases,cells); rows=[]
    for case in cases:
        reference=None
        for cell in [x for x in cells if x['case_id']==case.case_id]:
            method,seed,budget=cell['method'],cell['seed'],cell['budget']
            key=f'{case.case_id}__{method}__s{seed}__b{budget}'
            path=out/'raw'/(key+'.json.gz')
            if path.exists(): record=read(path); assert record['source_id']==sid
            else:
                cpu,wall=time.process_time(),time.perf_counter()
                result=(batched_exact(case,method) if method.startswith('batched') else
                        greedy(case,seed,budget) if method=='joint_greedy' else marginal(case,seed,budget,method))
                record=dict(key=key,**cell,source_id=sid,result=result,status='pass',
                    policy_cpu_seconds=time.process_time()-cpu,policy_wall_seconds=time.perf_counter()-wall,
                    target_version=case.target_version,oracle_version=case.oracle_version,metadata=case.metadata)
                save(path,record,True)
            if reference is None: reference=load_pinned_reference(case)
            canonical='left_first' if method=='batched_left' else 'right_first' if method=='batched_right' else 'count_allocation_feedback_audit_degree'
            record['validation']=strict_check_record(record,case,record_path=path,expected_key=key,expected_method=method,
                expected_seed=seed,expected_budget=budget,expected_source_id=sid,reference=reference,
                canonical_execution_method=canonical)
            result=record['result']; ledger=result['oracle_ledger']
            assert [i for b in result['batches'] for i in b['ids']]==[r['id'] for r in ledger]
            assert all(0<b['size']<=128 for b in result['batches'])
            save(path,record,True)
            summary=result['summary']
            row=dict(**cell,requests=summary['oracle_requests'],tokens_in=summary['oracle_tokens_in'],
                tokens_out=summary['oracle_tokens_out'],phase_counts=summary['phase_counts'],
                batches=len(result['batches']),batch_phases={p:sum(b['phase']==p for b in result['batches']) for p in ('audit','execution')},
                cpu_seconds=record['policy_cpu_seconds'],wall_seconds=record['policy_wall_seconds'],
                timings=result['timings'],loss=result['output']['loss'],full_count=result['output']['full_count'],
                sql_timings={k:result['output'][k] for k in ('sql_preprocessing_seconds','fact_ingest_seconds',
                    'returned_materialization_seconds','reference_materialization_seconds','reference_label_loading_seconds','sql_validation_seconds')},
                coverage=record['validation'].get('joint_count_coverage_observed'),
                conflict=result.get('conflict_freeze',False),raw_sha256=sha(path))
            rows.append(row); save(out/'progress.json',dict(expected=len(cells),completed=len(rows),rows=rows))
            print(json.dumps(dict(completed=len(rows),expected=len(cells),**row),default=convert),flush=True)
        case.close_sql()
    save(out/'summary.json',dict(source_id=sid,expected=len(cells),verified=len(rows),rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=['constructed','native'],required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--pilot',action='store_true')
    run(parser.parse_args())
