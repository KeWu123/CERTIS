"""Add method-semantic charge/phase checks to the unchanged first verifier.

Historical and currently running studies retain their frozen verifier.
This post-execution extension is independently applicable to their immutable
records. It never supplies reference truth to a policy.
"""
from pathlib import Path
from fractions import Fraction
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'code'))
from strict_result_validation import (StrictValidationError, load_pinned_reference,
    PinnedReference, ORIGINAL_METHOD_ALIASES, strict_check_record as _base_check)
from common import possible_unknowns, make_join
from bilinear_certificate import RegionSide, key_support, InconsistentEvidence


EXACT_NO_HELPER={'exact_scan','left_first','right_first','dynamic_max_degree'}
EXACT_HELPER={'helper_greedy','helper_negative_first'}
MECHANISMS={'keywise_degree','fixed_mask_pruning_degree','separate_extrema_degree'}


def _require(condition,code,message):
    if not condition:raise StrictValidationError(code,message)


def execution_contract(method):
    """Known fixed-study method contracts; unknown names fail closed.

    A runner with a wrapper name must explicitly name a canonical contract,
    independently of the saved record. Free-form aliases do not grant a
    different charging or phase contract.
    """
    method=ORIGINAL_METHOD_ALIASES.get(method,method)
    if method in EXACT_NO_HELPER:return {'execution'},False
    if method in EXACT_HELPER:return {'execution'},True
    if method=='cover':return {'cut','execution'},False
    if method in MECHANISMS:return {'audit','execution'},False
    if method in ('cut_feedback_audit','cut_static_audit','cut_feedback_point','cut_static_point'):
        return {'cut','audit','execution'},True
    direct_names={prefix+fb+'_'+source+'_'+ranking
        for prefix in ('direct_','keywise_')
        for fb in ('feedback','static') for source in ('audit','point')
        for ranking in ('degree','helper')}
    direct_names.update('count_allocation_'+fb+'_'+source+'_degree'
        for fb in ('feedback','static') for source in ('audit','point'))
    if method in direct_names:
        return {'audit','execution'},method.endswith('_helper')
    raise StrictValidationError('contract.unknown_method','Explicit supported execution contract required: '+str(method))


def _symbolic_affine(case,result,ledger,canonical):
    """Rebuild every coefficient from incidences, not one full-truth scalar."""
    cut_facts={r['id']:r['label'] for r in ledger if r['phase']=='cut'}
    residual=make_join(case).linearize(cut_facts)
    expected_weights=residual.weight_map()
    if canonical=='cover':
        d=result['diagnostics']
        weights={int(j):int(w) for j,w in d['residual_weights']}
        expected=expected_weights
        constant=d['residual_constant']
    else:
        weights={int(j):int(w) for j,w in result['affine_weights'].items()}
        expected={j:expected_weights.get(j,0) for j in range(case.n) if j not in cut_facts}
        constant=result['affine_constant']
    _require(weights==expected and constant==residual.constant,
             'contract.symbolic_affine','Coefficients differ from the full metadata-derived affine identity')


def _final_support(case,result,facts,discarded,canonical):
    if canonical in EXACT_NO_HELPER|EXACT_HELPER|{'cover'}:return
    try:
        if canonical.startswith('cut_'):
            weights={int(j):int(w) for j,w in result['affine_weights'].items()}
            value=-result['affine_constant']-sum(w*facts[j] for j,w in weights.items() if j in facts)
            for region in result['regions']:
                unknown=[j for j in region['ids'] if j not in facts]
                h=sum(facts.get(j,0) for j in region['ids'])
                lo,hi=max(0,region['lower']-h),min(len(unknown),region['upper']-h)
                if lo>hi:raise InconsistentEvidence('Empty affine bin')
                coefficients=sorted((weights.get(j,0)*(10*int(j in discarded)-1) for j in unknown),reverse=True)
                count=min(hi,max(lo,sum(c>0 for c in coefficients)))
                value+=sum(coefficients[:count])
        else:
            intervals={tuple(r['ids']):(r['lower'],r['upper']) for r in result['regions']}
            local=[];lost_max=total_min=0
            for left,right in zip(case.key_left,case.key_right):
                sides=[]
                for ids in (left,right):
                    unknown=[j for j in ids if j not in facts]
                    h=sum(facts.get(j,0) for j in ids)
                    d=sum(j in discarded for j in unknown)
                    lo,hi=intervals.get(ids,(0,0))
                    sides.append(RegionSide(lo,hi,h,d,len(unknown)-d))
                local.append(key_support(*sides,Fraction(1,10)))
                if canonical=='separate_extrema_degree':
                    lost_max+=key_support(*sides,Fraction(0))
                    total_min+=max(sides[0].lower_total,sides[0].known_positive)*max(sides[1].lower_total,sides[1].known_positive)
            if canonical=='separate_extrema_degree':value=10*lost_max-total_min
            elif canonical.startswith('keywise_') or canonical=='keywise_degree':value=max(local,default=0)
            else:value=sum(local)
    except InconsistentEvidence:
        _require(result.get('final_scaled_support') is None,'contract.final_support','Empty evidence reported a support')
        return
    _require(result.get('final_scaled_support')==value and value<=0,
             'contract.final_support','Reported final certificate differs from recomputed integer support')


def strict_check_record(record,case,*,canonical_execution_method=None,**kwargs):
    """First verify explicit cell/tape binding, then policy-semantic fields."""
    checked=_base_check(record,case,**kwargs)
    canonical=canonical_execution_method or kwargs['expected_method']
    canonical=ORIGINAL_METHOD_ALIASES.get(canonical,canonical)
    phases,helper_required=execution_contract(canonical)
    result=record['result'];ledger=result.get('oracle_ledger',result.get('ledger'))
    facts={int(j):y for j,y in result['facts'].items()}
    discarded=result.get('discarded',[])
    _require(isinstance(discarded,list) and all(type(j) is int and 0<=j<case.n for j in discarded)
             and len(discarded)==len(set(discarded)),
             'contract.discarded','Distinct in-target unknown discarded identities required')
    discarded=set(discarded)
    is_exact=canonical in EXACT_NO_HELPER|EXACT_HELPER|{'cover'}
    if is_exact:
        _require(not discarded,'contract.exact_mask','Exact controls cannot discard unknown identities')
        _require(result['output']['lost_count']==0 and result['output'].get('exact') is True,
                 'contract.exact_output','Exact controls must return the complete true bag')
    else:
        _require(result.get('allowance')=='1/10' and result.get('delta')=='1/20',
                 'contract.quality_parameters','Recorded parameters differ from the fixed study contract')
    retained=np.ones(case.n,dtype=bool)
    retained[list(discarded)]=False
    _require(not possible_unknowns(case,facts,retained),'contract.retained_completion',
             'Unresolved possible retained identities remain despite reported output completion')
    if canonical=='exact_scan':
        required=list(possible_unknowns(case,{}))
        _require([r['id'] for r in ledger]==required,'contract.scan_domain',
                 'Scan must buy its complete ordered initial structural domain')
    _require(all(r['phase'] in phases for r in ledger),'contract.oracle_phase',
             'Charged phase is impossible for the declared executor')
    if kwargs['expected_budget']==0:
        _require(not any(r['phase']=='audit' for r in ledger),'contract.zero_audit',
                 'A zero-budget cell contains random audit acquisitions')
    if canonical=='cover':
        planned=result.get('diagnostics',{}).get('cut_acquire')
        _require(isinstance(planned,list),'contract.cover_cut_plan','Recorded cut identities required')
        actual_cut=[r['id'] for r in ledger if r['phase']=='cut']
        _require(actual_cut==planned,'contract.cover_cut_identity',
                 'Per-ID cut phases differ from the recorded structural acquisition')
        _require([r['phase'] for r in ledger]==['cut']*len(planned)+['execution']*(len(ledger)-len(planned)),
                 'contract.cover_cut_prefix','Cover acquisitions must precede exact residual execution')
    if canonical=='cover' or canonical.startswith('cut_'):
        _symbolic_affine(case,result,ledger,canonical)
    _final_support(case,result,facts,discarded,canonical)
    actual_helper={r['id'] for r in result['helper_ledger']}
    expected_helper=set(range(case.n)) if helper_required else set()
    _require(actual_helper==expected_helper,'contract.helper_scope',
             'Helper-dependent methods buy every target score; metadata methods buy none')
    # These redundant descriptive fields must not contradict the cryptographic
    # case/version binding even when an old runner legitimately omitted them.
    expected_fields={'predicate_left':case.predicate_left,'predicate_right':case.predicate_right,
        'dataset':case.dataset,'variant':case.variant,'n':case.n,'key_count':len(case.key_names)}
    for container_name,container in (('record',record),('result',result)):
        for field,expected in expected_fields.items():
            if field in container:
                _require(container[field]==expected,'contract.'+container_name+'.'+field,
                         'Descriptive target metadata contradicts expected case')
    checked=dict(checked)
    checked.update(method_semantic_phases_verified=True,complete_helper_scope_verified=True,
                   descriptive_predicate_metadata_verified=True,
                   retained_completion_verified=True,final_integer_support_recomputed=not is_exact,
                   symbolic_affine_identity_verified=canonical=='cover' or canonical.startswith('cut_'),
                   canonical_execution_method=canonical)
    return checked
