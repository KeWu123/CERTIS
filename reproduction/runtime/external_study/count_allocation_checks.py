"""Boolean masks and actual SQL checks for the all-mask execution driver."""
from itertools import product
from pathlib import Path
import hashlib,json,random,time
from common import make_tiny_case
from count_allocation_arms import CountEnvelope,run_count_allocation,eligible


def pairs(case,z,discard=()):
    return {(k,i,j) for k,(left,right) in enumerate(zip(case.key_left,case.key_right))
            for i in left for j in right if z[i] and z[j] and i not in discard and j not in discard}


def main():
    rng=random.Random(2610163);mask_checks=0;sql_checks=0
    for repeat in range(150):
        truth=[rng.randrange(2) for _ in range(6)]
        case=make_tiny_case(side=(0,0,0,1,1,1),key_left=((0,1),(2,),()),
            key_right=((3,),(4,),()),truth=truth,name='count_check')
        facts={j:y for j,y in enumerate(truth) if rng.randrange(3)==0}
        intervals={}
        for ids in (*case.key_left,*case.key_right):
            total=sum(truth[j] for j in ids)
            intervals[ids]=(rng.randrange(total+1),rng.randrange(total,len(ids)+1))
        old=set() if repeat%2 else {j for j in range(case.n) if j not in facts and rng.randrange(4)==0}
        worlds=[z for z in product((0,1),repeat=6)
                if all(z[j]==y for j,y in facts.items()) and
                all(lo<=sum(z[j] for j in ids)<=hi for ids,(lo,hi) in intervals.items())]
        best=-1
        for bits in product((0,1),repeat=6):
            mask={j for j,b in enumerate(bits) if b}
            if mask & facts.keys() or not old<=mask:continue
            worst=max(9*len(pairs(case,z))-10*len(pairs(case,z,mask)) for z in worlds)
            if worst<=0:best=max(best,len(mask))
        env=CountEnvelope(case,intervals)
        try:mask,plan=env.plan(facts,old)
        except Exception as exc:
            from bilinear_certificate import InconsistentEvidence
            assert isinstance(exc,InconsistentEvidence) and best==-1
        else:
            assert len(mask)==best and old<=mask
            assert env.support(facts,mask)<=0
        mask_checks+=1;case.close_sql()
    for truth in product((0,1),repeat=6):
        case=make_tiny_case(side=(0,0,0,1,1,1),key_left=((0,1),(2,)),
            key_right=((3,4),(5,)),truth=truth,name='count_execute')
        for audit in (False,True):
            for feedback in (False,True):
                result=run_count_allocation(case,101,1,audit=audit,feedback=feedback)
                returned=case._sql['connection'].execute('SELECT key_id,left_id,right_id FROM returned_derivations').fetchall()
                assert set(returned)==pairs(case,truth,result['discarded'])
                assert len(returned)==result['output']['returned_count']
                assert result['summary']['helper_requests']==0
                assert len(result['oracle_ledger'])==len(result['facts'])==result['summary']['oracle_requests']
                assert {r['id'] for r in result['oracle_ledger'] if r['phase']=='execution'}<=set(result['initial_retained_unknown'])
                covered=all(r['lower']<=sum(truth[j] for j in r['ids'])<=r['upper'] for r in result['regions'])
                if covered:assert 10*result['output']['lost_count']<=result['output']['full_count']
                if not feedback:assert result['initial_discarded']==result['discarded']
                sql_checks+=1
        case.close_sql()
    repeated=make_tiny_case(side=(0,0,1,1),key_left=((0,1),(0,)),
                           key_right=((2,),(3,)),truth=(1,0,1,0),name='reject_reuse')
    assert not eligible(repeated)
    try:CountEnvelope(repeated,{})
    except ValueError:pass
    else:raise AssertionError('Repeated identities accepted')
    batch_checks=[]
    for density in (0.,.1,.5,1.):
        truth=[int(rng.random()<density) for _ in range(300)]
        case=make_tiny_case(side=[0]*150+[1]*150,key_left=(tuple(range(150)),),
            key_right=(tuple(range(150,300)),),truth=truth,name='count_batches')
        r=run_count_allocation(case,202,5)
        assert 10*r['output']['lost_count']<=r['output']['full_count']
        assert {a['id'] for a in r['oracle_ledger'] if a['phase']=='execution'}<=set(r['initial_retained_unknown'])
        batch_checks.append(dict(density=density,events=len(r['events']),requests=r['summary']['oracle_requests']))
        sql_checks+=1;case.close_sql()
    out=dict(status='pass',independent_boolean_mask_checks=mask_checks,actual_sql_executions=sql_checks,
             repeated_identity_rejected=True,multiple_batch_cases=batch_checks,
             source_sha256=hashlib.sha256(Path(__file__).with_name('count_allocation_arms.py').read_bytes()).hexdigest())
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':main()
