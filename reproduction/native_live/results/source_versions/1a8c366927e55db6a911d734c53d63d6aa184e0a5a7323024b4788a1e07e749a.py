"""Post-matrix exact all-mask planner; eligible key-disjoint targets only."""
from fractions import Fraction
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'code'))
from count_mask_planner import SideCounts, plan_counts
from bilinear_certificate import InconsistentEvidence, RegionSide, key_support
from common import Oracle, evaluate, potential_degrees, possible_unknowns
from direct_arms import make_regions, ALLOWANCE, DELTA


def eligible(case):
    membership=np.zeros(case.n,dtype=np.int64)
    for left,right in zip(case.key_left,case.key_right):
        membership[list(left)]+=1;membership[list(right)]+=1
    return bool(np.all(membership<=1))


class CountEnvelope:
    def __init__(self,case,intervals):
        if not eligible(case):
            raise ValueError('All-mask count allocation requires key-disjoint identities')
        self.case,self.intervals=case,intervals
        self.incident=set(j for pair in zip(case.key_left,case.key_right) for ids in pair for j in ids)
        self.calls=0

    def plan(self,facts,old):
        self.calls+=1
        old=set(old)-facts.keys()
        metadata,groups=[],[]
        for pair in zip(self.case.key_left,self.case.key_right):
            sides,unknowns=[],[]
            for ids in pair:
                unknown=tuple(j for j in ids if j not in facts)
                h=sum(facts.get(j)==1 for j in ids)
                lo,hi=self.intervals.get(ids,(0,0))
                sides.append(SideCounts(lo,hi,h,len(unknown),len(old.intersection(unknown))))
                unknowns.append(unknown)
            metadata.append(tuple(sides));groups.append(tuple(unknowns))
        answer=plan_counts(metadata,ALLOWANCE)
        if answer is None:
            raise InconsistentEvidence('No certified superset of previous discards')
        # All orphan unknown identities can be omitted without losing a pair.
        mask=(set(range(self.case.n))-self.incident-facts.keys())|old
        for pair,counts in zip(groups,answer.counts):
            for unknown,count in zip(pair,counts):
                forced=old.intersection(unknown)
                mask.update(forced)
                additions=[j for j in unknown if j not in forced][:count-len(forced)]
                mask.update(additions)
        if not old<=mask or mask & facts.keys():
            raise AssertionError('Identity containment/reconstruction failed')
        if len(mask)!=answer.discarded+len(set(range(self.case.n))-self.incident-facts.keys()):
            raise AssertionError('Count plan and identity mask disagree')
        support=self.support(facts,mask)
        if support!=answer.support or support>0:
            raise AssertionError('Independent original support rejects allocation')
        return mask,answer

    def support(self,facts,mask):
        support=0
        for pair in zip(self.case.key_left,self.case.key_right):
            sides=[]
            for ids in pair:
                unknown=[j for j in ids if j not in facts]
                h=sum(facts.get(j)==1 for j in ids);d=len(mask.intersection(unknown))
                lo,hi=self.intervals.get(ids,(0,0))
                sides.append(RegionSide(lo,hi,h,d,len(unknown)-d))
            support+=key_support(*sides,ALLOWANCE)
        return support


def run_count_allocation(case,seed,audit_budget,*,audit=True,feedback=True,materialize=True):
    start=time.perf_counter()
    oracle=Oracle(case)
    t=time.perf_counter()
    rank=potential_degrees(case,{})
    execution_order=np.lexsort((np.arange(case.n),-rank))
    prep=time.perf_counter()-t
    intervals,records,audit_seconds=make_regions(case,oracle,seed,audit_budget if audit else 0)
    envelope=CountEnvelope(case,intervals)
    t=time.perf_counter();mask,answer=envelope.plan(oracle.facts,set())
    projection=time.perf_counter()-t
    initial_mask=set(mask)
    initial_retained=set(range(case.n))-initial_mask-oracle.facts.keys()
    conflict=False;bookkeeping=0.
    events=[dict(paid=len(oracle.ledger),discard_cardinality=len(mask),scaled_support=answer.support,
                 optimal_counts=answer.counts,backend=answer.backend)]
    while True:
        t=time.perf_counter();retained=np.ones(case.n,dtype=bool);retained[list(mask)]=False
        possible=set(possible_unknowns(case,oracle.facts,retained))
        selected=[int(j) for j in execution_order if int(j) in possible][:128]
        bookkeeping+=time.perf_counter()-t
        if not selected:break
        oracle.acquire(selected,'execution')
        if set(selected)&mask:raise AssertionError('Acquired a previously discarded identity')
        if feedback and not conflict:
            t=time.perf_counter()
            try:mask,answer=envelope.plan(oracle.facts,mask)
            except InconsistentEvidence:conflict=True
            projection+=time.perf_counter()-t
        events.append(dict(paid=len(oracle.ledger),discard_cardinality=len(mask),
                           scaled_support=None if conflict else answer.support,
                           optimal_counts=None if conflict else answer.counts,conflict=conflict))
    algorithm=time.perf_counter()-start
    execution_ids={a['id'] for a in oracle.ledger if a['phase']=='execution'}
    if not execution_ids<=initial_retained:
        raise AssertionError('Same-start containment failed')
    retained=np.ones(case.n,dtype=bool);retained[list(mask)]=False
    if possible_unknowns(case,oracle.facts,retained):
        raise AssertionError('Retained materialization incomplete')
    try:final_support=envelope.support(oracle.facts,mask)
    except InconsistentEvidence:final_support=None
    return dict(method='count_allocation_'+('feedback' if feedback else 'static')+
                ('_audit' if audit else '_point')+'_degree',
        post_matrix_exploratory=True,preregistered=False,seed=seed,
        audit_budget=audit_budget if audit else 0,mode='single_key_exact',
        allowance=str(ALLOWANCE),delta=str(DELTA),summary=oracle.summary(),
        oracle_ledger=oracle.ledger,helper_ledger=oracle.helper_ledger,facts=oracle.facts,
        regions=records,discarded=sorted(mask),initial_discarded=sorted(initial_mask),
        initial_retained_unknown=sorted(initial_retained),request_containment_checked=True,
        conflict_freeze=conflict,final_scaled_support=final_support,
        initial_certified_discard_count=len(initial_mask),final_certified_discard_count=len(mask),
        projection_calls=envelope.calls,events=events,
        output=evaluate(case,oracle.facts) if materialize else None,
        timings=dict(preprocessing_seconds=prep,audit_seconds=audit_seconds,
                     projection_seconds=projection,execution_bookkeeping_seconds=bookkeeping,
                     algorithm_seconds=algorithm))
