"""Execute a fixed bag query while refining its ORIGINAL label evidence.

The callback sees only labels actually acquired. It must use an unchanged
confidence event; execution labels are point facts, never extra random draws.
Underlying discard masks expand monotonically. No label is requested outside
the original retained set. Statistical correctness is on the original event.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import numbers
import time

import numpy as np


@dataclass
class ExecutionResult:
    facts: dict[int, int]
    request_ids: tuple[int, ...]
    initial_index: int
    final_index: int
    ledger: list[dict]
    projection_seconds: float
    execution_seconds: float
    confidence_conflict: bool


def execute_refining(weights, masks, initial_facts, order, oracle, prepare,
                     decide, *, batch_size=512, feedback=True):
    """Largest accepted nested mask, followed by optional execution feedback.

    prepare(facts) returns an evidence state; decide(state, index) returns an
    exact Boolean acceptance. Both callbacks are denied unknown labels.
    On an evidence conflict, keep the last accepted policy and finish its
    execution, recording the conflict. Never reinterpret execution as sampling.
    The original probability statement allows failure off its covered event.
    """
    w = tuple(weights)
    n = len(w)
    if n < 1:
        raise ValueError("A nonempty finite target is required")
    if any(not isinstance(v, numbers.Integral) or v < 0 for v in w):
        raise ValueError("Nonnegative integer bag multiplicities required")
    w = tuple(int(v) for v in w)
    if not isinstance(batch_size, numbers.Integral) or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    family = tuple(np.asarray(a).copy() for a in masks)
    if not family or any(a.dtype != np.bool_ or a.shape != (n,) for a in family):
        raise ValueError("A Boolean mask per identity is required")
    if family[0].any() or any(np.any(a & ~b) for a,b in zip(family,family[1:])):
        raise ValueError("Masks must be nested, starting with empty discard")
    for a in family:
        a.setflags(write=False)
    permutation = tuple(order)
    if (len(permutation) != n or any(not isinstance(j,numbers.Integral) for j in permutation)
            or set(int(j) for j in permutation) != set(range(n))):
        raise ValueError("Execution order must be a target permutation")
    permutation = tuple(int(j) for j in permutation)
    facts = {}
    for j,y in initial_facts.items():
        if not isinstance(j,numbers.Integral) or not 0 <= j < n or y not in (0,1):
            raise ValueError("Truthful in-target binary point facts required")
        facts[int(j)] = int(y)
    initial = dict(facts)
    known = np.zeros(n,dtype=bool); known[list(facts)] = True
    positive_weight = np.array([v>0 for v in w],dtype=bool)
    requests = []; ledger = []; index = 0; projection = 0.; execution = 0.
    conflict = False
    def advance(current):
        nonlocal projection,conflict
        start = time.perf_counter()
        state = prepare(dict(facts))
        if not state['consistent']:
            conflict = True
        else:
            while current+1 < len(family) and decide(state,current+1):
                current += 1
        projection += time.perf_counter()-start
        return current
    index = advance(index)
    first = index
    initially_required = ~family[first] & ~known & positive_weight
    while True:
        eligible = ~family[index] & ~known & positive_weight
        batch = [j for j in permutation if eligible[j]][:int(batch_size)]
        if not batch:
            break
        start = time.perf_counter()
        labels = tuple(oracle(tuple(batch)))
        if len(labels) != len(batch) or any(y not in (0,1) for y in labels):
            raise ValueError("Oracle must return one immutable binary label per requested identity")
        for j,y in zip(batch,labels):
            if j in facts or not initially_required[j]:
                raise ArithmeticError("Repeated or originally discarded label request")
            facts[j] = int(y); known[j] = True; requests.append(j)
        execution += time.perf_counter()-start
        previous = index
        if feedback and not conflict:
            index = advance(index)
        if index < previous:
            raise ArithmeticError("Underlying discard frontier shrank")
        ledger.append({'batch':len(ledger)+1,'new_requests':list(batch),
                       'labels':[int(y) for y in labels], 'frontier_before':previous,
                       'frontier_after':index,'known_count':len(facts),
                       'confidence_conflict':conflict})
    assert len(requests)==len(set(requests))
    assert set(initial.items()) <= set(facts.items())
    assert all(initially_required[j] for j in requests)
    if not feedback:
        start=time.perf_counter()
        conflict=conflict or not prepare(dict(facts))['consistent']
        projection+=time.perf_counter()-start
    return ExecutionResult(facts,tuple(requests),first,index,ledger,projection,execution,conflict)


class PreparedAudit:
    """Incremental geometry of a canonical build_prefix_audit record.

    Count endpoints and original sample identities are never modified. Atom
    prefix sums make cache refinement O(N + H*grid), rather than rescanning
    all identities separately for every old inequality. As in the producer,
    scores/grid lie in [0,1], the grid starts at 0, and strata are 0,...,H-1.
    Broader arbitrary JointEvidence inputs use the original core API directly.
    """
    def __init__(self,audit,core):
        self.audit,self.core = audit,core
        self.size = len(audit.scores)
        s=np.asarray(audit.scores); h=np.asarray(audit.strata)
        sampled=np.asarray(audit.sampled); observed=np.asarray(audit.observed)
        grid=np.asarray(audit.grid)
        if (self.size<1 or s.ndim!=1 or h.shape!=(self.size,)
                or sampled.shape!=(self.size,) or observed.shape!=(self.size,)
                or sampled.dtype!=np.bool_ or np.any(~np.isfinite(s))
                or np.any((s<0)|(s>1)) or not np.issubdtype(h.dtype,np.integer)
                or not np.array_equal(np.unique(h),np.arange(int(h.max())+1))
                or grid.ndim!=1 or len(grid)==0 or grid[0]!=0
                or np.any(~np.isfinite(grid)) or np.any((grid<0)|(grid>1))
                or np.any(np.diff(grid)<=0) or len(audit.intervals)!=len(grid)
                or np.any((observed[sampled]!=0)&(observed[sampled]!=1))):
            raise ValueError("Malformed original audit dimensions, scores, strata, grid or point labels")
        self.version = (audit.target_version,audit.oracle_version)
        if any(not isinstance(v,str) or not v for v in self.version):
            raise ValueError("Explicit immutable target and truth versions required")
        self.initial = {int(j):int(audit.observed[j]) for j in np.flatnonzero(audit.sampled)}
        self.bin_ids = np.searchsorted(audit.grid,audit.scores,side='right')
        self.members = [tuple(np.flatnonzero(audit.strata==h)) for h in np.unique(audit.strata)]
        self.atom_ids = [tuple(tuple(int(j) for j in ids if self.bin_ids[j]==b)
                               for b in range(len(audit.grid)+1)) for ids in self.members]
        self.regions = []
        for h in np.unique(audit.strata):
            rows=[]
            for gi,regions in enumerate(audit.intervals):
                relevant=[x for x in regions if x['stratum']==h]
                if len(relevant)!=2 or {bool(x['lost']) for x in relevant}!={True,False}:
                    raise ValueError("Missing prefix/suffix audit constraint")
                for x in relevant:
                    lo,hi=x['lower'],x['upper']
                    if int(lo)!=lo or int(hi)!=hi:
                        raise ValueError("Integer count endpoints required")
                    rows.append((gi,bool(x['lost']),int(lo),int(hi)))
            self.regions.append(tuple(rows))

    def refine(self,facts):
        if any(facts.get(j)!=y for j,y in self.initial.items()):
            raise ValueError("Original sampled labels changed or were removed")
        if any(not isinstance(j,numbers.Integral) or not 0<=j<self.size or y not in (0,1)
               for j,y in facts.items()):
            raise ValueError("Invalid cache facts")
        positives=tuple(sorted(j for j,y in facts.items() if y))
        negatives=tuple(sorted(j for j,y in facts.items() if not y))
        chains=[]; original=[]; consistent=True
        for atoms,regions in zip(self.atom_ids,self.regions):
            unknown=tuple(tuple(j for j in atom if j not in facts) for atom in atoms)
            pos=[sum(facts.get(j,0) for j in atom) for atom in atoms]
            ps=np.cumsum(pos).tolist(); ms=np.cumsum([len(a) for a in unknown]).tolist()
            bounds=[]
            for gi,prefix,lo,hi in regions:
                k,m=(ps[gi],ms[gi]) if prefix else (ps[-1]-ps[gi],ms[-1]-ms[gi])
                lower,upper=max(0,lo-k),min(m,hi-k)
                bounds.append((gi,prefix,lower,upper))
                if lower>upper:consistent=False
            chains.append(self.core.Chain(unknown,tuple(bounds)))
            original.append((atoms,unknown,bounds))
        if consistent:
            try:
                for chain in chains:
                    graph=self.core._graph(chain,self.core._groups(chain))
                    self.core.nx.single_source_bellman_ford_path_length(graph,0,weight='weight')
            except self.core.nx.NetworkXUnbounded:
                consistent=False
        ev=self.core.JointEvidence(*self.version,self.size,positives,negatives,
                                   tuple(chains),consistent,None if consistent else 'count conflict')
        return {'joint':ev,'consistent':consistent,'facts':facts,'geometry':original}


def disjoint_support(regions,weights,discarded,facts,allowance):
    """Independent exact signed support over disjoint residual count regions."""
    r=Fraction(allowance); p,q=int(r.numerator),int(r.denominator)
    value=-p*sum(int(weights[j]) for j,y in facts.items() if y)
    for ids,lo,hi in regions:
        if lo>hi or lo<0 or hi>len(ids):
            raise ArithmeticError("Inconsistent disjoint region")
        coefficients=sorted((int(weights[j])*(q*int(discarded[j])-p) for j in ids),reverse=True)
        count=max(int(lo),min(int(hi),sum(c>0 for c in coefficients)))
        value+=sum(coefficients[:count])
    return value


def decision_callbacks(prepared,weights,masks,mode,allowance=Fraction(1,10),bin_counts=None):
    """Identical feedback interface for joint, strongest-single, bins, point-only."""
    if mode not in ('joint','best_single','bins','point_only'):
        raise ValueError("Unknown evidence mode")
    core=prepared.core; w=core.exact_weights(weights,prepared.size); r=Fraction(allowance)
    r=Fraction(int(r.numerator),int(r.denominator))
    if not 0<=r<=1:
        raise ValueError("Allowance in [0,1] required")
    def prepare(facts):
        if mode in ('joint','best_single'):
            state=prepared.refine(facts)
            if mode=='best_single':
                # An empty conjunction does not imply empty individual sets.
                # Each inconsistent single set uses its own point-fact fallback.
                state['consistent']=True
            return state
        regions=[]; consistent=True
        if mode=='bins':
            for ids,lo,hi in bin_counts:
                pos=sum(facts.get(j,0) for j in ids); unknown=tuple(j for j in ids if j not in facts)
                lo,hi=max(0,lo-pos),min(len(unknown),hi-pos)
                if lo>hi:consistent=False
                regions.append((unknown,lo,hi))
        return {'consistent':consistent,'facts':facts,'regions':regions}
    def decide(state,index):
        mask=masks[index]; facts=state['facts']
        if mode=='joint':
            return core.decision_joint_mask(state['joint'],w,mask,r,*prepared.version)['accepted']
        if mode=='point_only':
            lost=sum(w[j] for j in range(len(w)) if mask[j] and j not in facts)
            known=sum(w[j] for j,y in facts.items() if y)
            return (r.denominator-r.numerator)*lost <= r.numerator*known
        if mode=='bins':
            return disjoint_support(state['regions'],w,mask,facts,r)<=0
        for gi in range(len(prepared.audit.grid)):
            regions=[]
            for _,atoms,bounds in state['geometry']:
                for boundary,prefix,lo,hi in bounds:
                    if boundary!=gi:continue
                    ids=tuple(j for atom in (atoms[:gi+1] if prefix else atoms[gi+1:]) for j in atom)
                    regions.append((ids,lo,hi))
            if any(lo>hi for ids,lo,hi in regions):
                lost=sum(w[j] for j in range(len(w)) if mask[j] and j not in facts)
                known=sum(w[j] for j,y in facts.items() if y)
                if (r.denominator-r.numerator)*lost<=r.numerator*known:return True
                continue
            if disjoint_support(regions,w,mask,facts,r)<=0:return True
        return False
    return prepare,decide
