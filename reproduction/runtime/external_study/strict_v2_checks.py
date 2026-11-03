"""Differentiated method-semantic corruption probes on original raw records."""
from pathlib import Path
from copy import deepcopy
import gzip,json,sys,hashlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import load_cases
from strict_result_validation_v2 import strict_check_record,load_pinned_reference,StrictValidationError


def reconcile(r):
    result=r['result'];s=result['summary'];ledger=result.get('oracle_ledger',result.get('ledger'))
    counts={p:sum(x['phase']==p for x in ledger) for p in ('cut','audit','execution')}
    tokens={p:{d:sum(x[d] for x in ledger if x['phase']==p) for d in ('tokens_in','tokens_out')}
            for p in ('cut','audit','execution')}
    for container in (s,result):
        if 'phase_counts' in container:container['phase_counts']=deepcopy(counts)
        if 'phase_tokens' in container:container['phase_tokens']=deepcopy(tokens)


def main():
    sid='db17c4543b8832d2acf2fc3db26099461a7055dc1ed64ad0867c0e5a47c8c22f'
    folder=HERE/'results'/sid/'raw';case=load_cases()[0];reference=load_pinned_reference(case)
    records={}
    for method in ('exact_scan','helper_greedy','left_first','right_first','cover'):
        key=f'{case.case_id}__{method}__s0__b0';path=folder/(key+'.json.gz')
        r=json.loads(gzip.decompress(path.read_bytes()))
        args=dict(record_path=path,expected_key=key,expected_method=method,expected_seed=0,
                  expected_budget=0,expected_source_id=sid,reference=reference)
        strict_check_record(r,case,**args);records[method]=(r,args)
    rejected=[]
    def reject(name,r,args):
        try:strict_check_record(r,case,**args)
        except StrictValidationError as exc:rejected.append(dict(name=name,code=exc.code))
        else:raise AssertionError('Corruption accepted: '+name)
    r,args=records['exact_scan'];bad=deepcopy(r)
    bad['result']['ledger'][0]['phase']='cut';reconcile(bad)
    reject('self-consistent exact phase recategorization',bad,args)
    r,args=records['helper_greedy'];bad=deepcopy(r)
    bad['result']['helper_ledger']=[]
    for container in (bad['result'],bad['result']['summary']):
        for k in ('helper_requests','helper_tokens_in','helper_tokens_out','helper_read_items','helper_read_calls'):
            if k in container:container[k]=0
    reject('erase helper charge and reconcile all aggregates',bad,args)
    for field in ('predicate_left','predicate_right','dataset','variant','n','key_count'):
        r,args=records['exact_scan'];bad=deepcopy(r)
        bad[field]='incorrect' if isinstance(bad.get(field),str) else -1
        reject('target description '+field,bad,args)
    r,args=records['cover']
    for original,new in (('cut','execution'),('execution','cut')):
        bad=deepcopy(r)
        entry=next(x for x in bad['result']['ledger'] if x['phase']==original)
        entry['phase']=new;reconcile(bad)
        reject('cover per-ID '+original+' to '+new,bad,args)
    from strict_result_validation_v2 import execution_contract
    for method in ('count_allocation_feedback_point_helper','count_allocation_static_audit_helper','unregistered_policy'):
        try:execution_contract(method)
        except StrictValidationError as exc:rejected.append(dict(name='unknown contract '+method,code=exc.code))
        else:raise AssertionError('Unknown policy accepted')
    out=dict(status='pass',genuine_original_cells_accepted=len(records),mutations_rejected=rejected,
             v2_sha256=hashlib.sha256(Path(__file__).with_name('strict_result_validation_v2.py').read_bytes()).hexdigest())
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':main()
