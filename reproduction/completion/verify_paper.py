"""Verify the full archived experiment roster and independently reaggregate bags."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/study'),
    str(ROOT/'reproduction/runtime/external_study'),str(ROOT/'code')]
import common


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def validate(record,case,truth):
    result=record['result'];summary=result.get('summary',result)
    ledger=result.get('oracle_ledger',result.get('ledger'))
    facts={int(j):int(y) for j,y in result['facts'].items()}
    assert len(ledger)==len({r['id'] for r in ledger})==summary['oracle_requests']
    assert {r['id']:r['label'] for r in ledger}==facts
    assert all(int(truth[r['id']])==r['label'] for r in ledger)
    for direction in ('in','out'):
        assert sum(r['tokens_'+direction] for r in ledger)==summary['oracle_tokens_'+direction]
    for phase in ('audit','cut','execution'):
        assert sum(r['phase']==phase for r in ledger)==summary['phase_counts'][phase]
    full=sum(sum(int(truth[j]) for j in a)*sum(int(truth[j]) for j in b)
             for a,b in zip(case.key_left,case.key_right))
    returned=sum(sum(facts.get(j)==1 for j in a)*sum(facts.get(j)==1 for j in b)
                 for a,b in zip(case.key_left,case.key_right))
    output=result['output']
    assert output['full_count']==full and output['returned_count']==returned
    assert output['lost_count']==full-returned and output['bag_subset']
    assert 'returned_derivations' in output['returned_sql']
    coverage=True
    for region in result.get('regions',[]):
        actual=sum(int(truth[j]) for j in region['ids'])
        coverage &= region['lower']<=actual<=region['upper']
    return dict(requests=len(ledger),full=full,returned=returned,coverage=coverage,
        loss_violation=10*(full-returned)>full,
        conflict=bool(result.get('conflict_freeze',False)))


def main():
    archive=ROOT/'reproduction/archive'
    cases={c.case_id:c for c in common.load_cases()}
    from native_discussion_cases import load_native_cases
    native={c.case_id:c for c in load_native_cases()}
    groups=[('primary',archive/'primary/raw',1376,cases),
        ('count_allocation',archive/'primary/count_allocation/raw',112,cases),
        ('mechanisms',archive/'primary/mechanism_controls/raw',288,cases),
        ('extra_exact',archive/'primary/extra_exact_controls/raw',64,cases),
        ('native_original',archive/'native/raw',216,native)]
    reports=[]
    for name,directory,expected,roster in groups:
        paths=sorted(directory.glob('*.json.gz'))
        assert len(paths)==expected,(name,len(paths),expected)
        rows=[]
        for path in paths:
            record=read(path);case=roster[record['case_id']]
            row=validate(record,case,common.Oracle(case)._validation_labels())
            rows.append(dict(case_id=case.case_id,method=record['method'],
                seed=record.get('seed',0),budget=record.get('budget',0),**row))
        report=dict(group=name,records=len(rows),loss_violations=sum(r['loss_violation'] for r in rows),
            coverage_failure_records=sum(not r['coverage'] for r in rows),rows=rows)
        reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    sys.path.insert(0,str(ROOT/'reproduction/runtime/structural_study'))
    import structural_cases
    specifications={s['target']:s for s in json.loads((archive/'structural/inputs.json').read_text())}
    paths=sorted((archive/'structural/raw').glob('*.json.gz'));assert len(paths)==594
    rows=[]
    for path in paths:
        record=read(path);row=record['row'];spec=specifications[row['target']]
        case=structural_cases.build(spec)
        checked=validate(record,case,spec['truth'])
        assert not (checked['coverage'] and checked['loss_violation'])
        rows.append(dict(target=row['target'],method=row['method'],**checked))
    reports.append(dict(group='structural',records=594,rows=rows,
        loss_violations=sum(r['loss_violation'] for r in rows)))
    geometry=json.loads((archive/'structural/geometry_records.json').read_text())
    assert len(geometry)==3168
    matched={}
    for method in ('fixed_grid','adaptive_grid','fixed_continuous','adaptive_continuous'):
        cohort='fixed_continuous_v2' if method=='fixed_continuous' else 'constructed_v1'
        paths=list((ROOT/'reproduction/study'/cohort/'raw').glob('*__'+method+'__s*__b500.json.gz'))
        assert len(paths)==48,(method,len(paths))
        matched[method]=sum(read(p)['result']['summary']['oracle_requests'] for p in paths)
    greedy=list((ROOT/'reproduction/runtime/greedy_count_profile/full/raw').glob(
        '*__greedy_count_profile_feedback_audit_degree__s*__b500.json.gz'))
    greedy=[p for p in greedy if p.name.startswith(('dbpedia__','goemotions__'))]
    assert len(greedy)==48
    matched['joint_greedy']=sum(read(p)['result']['summary']['oracle_requests'] for p in greedy)
    count=next(r for r in reports if r['group']=='count_allocation')['rows']
    matched['joint_dp']=sum(r['requests'] for r in count if r['budget']==500)
    separate=ROOT/'reproduction/completion/results/separate/summary.json'
    separate_rows=json.loads(separate.read_text(encoding='utf-8'))['rows']
    assert len(separate_rows)==48
    matched['separate_greedy']=sum(r['requests'] for r in separate_rows)
    expected=dict(fixed_grid=232680,adaptive_grid=232495,fixed_continuous=232549,
        adaptive_continuous=232371,separate_greedy=198807,joint_greedy=198142,joint_dp=197563)
    assert matched==expected,(matched,expected)
    report=dict(status='pass',groups=reports,geometry_records=len(geometry),matched_48_cell_totals=matched,
        note='Recomputed labels, paid ledgers, token/phase totals and factorized full/returned bag counts. '
             'Original records retain their materialized SQL validation evidence.')
    destination=ROOT/'outputs/paper_verification.json';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='pass',records=sum(r['records'] for r in reports),geometry_records=len(geometry))),flush=True)


if __name__=='__main__':main()
