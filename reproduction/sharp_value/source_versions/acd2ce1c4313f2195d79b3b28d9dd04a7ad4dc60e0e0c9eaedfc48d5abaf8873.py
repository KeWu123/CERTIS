"""Complete seed-101 native certificate ablation with exact snapshot diagnostics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from sharp_wrapper import RELEASE, HERE, WORKSPACE, execute, study
sys.path.insert(0, str(RELEASE/'reproduction/completion'))
from native_dbpedia import cases_for


def canonical(value):
    return json.dumps(json.loads(json.dumps(value,default=study.convert)), sort_keys=True,
        separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def audit_facts(result):
    return [row for row in result['oracle_ledger'] if row['phase']=='audit']


def aggregate(rows):
    by_query={}
    for r in rows:
        by_query.setdefault(r['case_id'], []).append(r['request_saving_percent'])
    query_savings=[statistics.mean(values) for values in by_query.values()]
    s=sum(r['separate_requests'] for r in rows)
    j=sum(r['joint_requests'] for r in rows)
    return dict(cells=len(rows),queries=len(by_query),separate_requests=s,joint_requests=j,
        requests_saved=s-j,pooled_saving_percent=100*(s-j)/s,
        wins=sum(r['request_gap']>0 for r in rows),ties=sum(r['request_gap']==0 for r in rows),
        losses=sum(r['request_gap']<0 for r in rows),
        query_macro_mean_saving_percent=statistics.mean(query_savings),
        query_median_saving_percent=statistics.median(query_savings),
        query_worst_saving_percent=min(query_savings),
        initial_joint_only_cells=sum(r['initial_joint_only'] for r in rows),
        initial_discard_gap_sum=sum(r['initial_discard_gap'] for r in rows),
        max_request_saving_percent=max(r['request_saving_percent'] for r in rows),
        min_request_saving_percent=min(r['request_saving_percent'] for r in rows))


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.chdir(RELEASE)
    cases=cases_for()
    if args.pilot:cases=cases[:1]
    files=[Path(__file__).resolve(),HERE/'sharp_wrapper.py',HERE/'PROTOCOL.md',
        RELEASE/'code/count_mask_planner.py', RELEASE/'code/bilinear_certificate.py',
        RELEASE/'code/piecewise_count_profile.py',RELEASE/'code/piecewise_numpy_profile.py',
        RELEASE/'reproduction/completion/native_dbpedia.py',
        RELEASE/'reproduction/runtime/external_study/count_allocation_arms.py',
        RELEASE/'reproduction/runtime/greedy_count_profile/greedy_planner.py']
    binding=dict(sources={str(p.relative_to(WORKSPACE)).replace('\\','/'):study.sha(p) for p in files},
        dataset_sha256=study.sha(study.common.SOURCE/'data/datasets/dbpedia.parquet'),
        arrays_sha256=study.sha(study.common.INPUTS/'dbpedia_arrays.npz'),
        targets={c.case_id:c.target_version for c in cases},seed=101,budgets=[125,500],
        methods=['joint','separate'],pilot=args.pilot)
    sid=digest(binding)
    manifest=dict(source_id=sid,**binding)
    frozen=out/'manifest.json'
    if frozen.exists():assert json.loads(frozen.read_text())==manifest
    else:study.save(frozen,manifest)
    rows=[]
    for case in cases:
        for budget in (125,500):
            records={}
            for method in ('joint','separate'):
                raw=out/'raw'/f'{case.case_id}__{method}__s101__b{budget}.json.gz'
                if raw.exists():
                    record=study.read(raw);assert record['source_id']==sid
                else:
                    case.close_sql();start=time.perf_counter()
                    result=execute(case,101,budget,method=method,instrument=True)
                    output=result['output'];truth=study.common.Oracle(case)._validation_labels()
                    coverage=all(r['lower']<=sum(int(truth[j]) for j in r['ids'])<=r['upper']
                        for r in result['regions'])
                    assert output['bag_subset']
                    assert not coverage or 10*output['lost_count']<=output['full_count']
                    assert not result['conflict_freeze']
                    validation=dict(bag_subset=True,coverage=coverage,
                        coverage_conditional_loss_check=True,conflict_freeze=False)
                    if method=='joint':
                        archive=RELEASE/'reproduction/completion/results/native_dbpedia/raw'/(
                            f'{case.case_id}__joint_greedy_cached__s101__b{budget}.json.gz')
                        old=study.read(archive)['result']
                        fields=('oracle_ledger','helper_ledger','facts','regions','discarded',
                            'initial_discarded','initial_retained_unknown','batches')
                        checks={field:digest(result[field])==digest(old[field]) for field in fields}
                        checks['sql_counts']=all(output[k]==old['output'][k] for k in
                            ('full_count','returned_count','lost_count','bag_subset'))
                        assert all(checks.values()),checks
                        validation['published_joint_equivalence']=dict(checks=checks,
                            archive=str(archive.relative_to(RELEASE)).replace('\\','/'),
                            archive_sha256=study.sha(archive))
                    record=dict(status='pass',source_id=sid,case_id=case.case_id,key=case.variant,
                        method=method,seed=101,budget=budget,target_version=case.target_version,
                        validation=validation,result=result,diagnostic_outer_seconds=time.perf_counter()-start)
                    study.save(raw,record,True)
                records[method]=record
                print(json.dumps(dict(completed=2*len(rows)+len(records),expected=4*len(cases),
                    case_id=case.case_id,budget=budget,method=method,
                    requests=record['result']['summary']['oracle_requests'])),flush=True)
            joint,separate=(records[m]['result'] for m in ('joint','separate'))
            assert digest(audit_facts(joint))==digest(audit_facts(separate))
            assert digest(joint['regions'])==digest(separate['regions'])
            for a,b in zip(joint['sharp_snapshots'][0]['keys'],separate['sharp_snapshots'][0]['keys']):
                assert a==b
            jr,sr=(r['summary']['oracle_requests'] for r in (joint,separate))
            initial=joint['sharp_snapshots'][0]
            row=dict(case_id=case.case_id,key=case.variant,budget=budget,seed=101,
                joint_requests=jr,separate_requests=sr,request_gap=sr-jr,
                request_saving_percent=100*(sr-jr)/sr,
                initial_joint_discard=joint['initial_certified_discard_count'],
                initial_separate_discard=separate['initial_certified_discard_count'],
                initial_discard_gap=initial['same_evidence_discard_gap'],
                initial_joint_only=initial['alternatives']['joint']['requires_coupling'],
                initial_joint_support=initial['alternatives']['joint']['joint_support'],
                initial_joint_mask_separate_support=initial['alternatives']['joint']['separate_support'],
                joint_loss=joint['output']['loss'],separate_loss=separate['output']['loss'],
                snapshots={m:len(records[m]['result']['sharp_snapshots']) for m in records},
                raw={m:dict(path=f'raw/{case.case_id}__{m}__s101__b{budget}.json.gz',
                    sha256=study.sha(out/'raw'/f'{case.case_id}__{m}__s101__b{budget}.json.gz')) for m in records})
            rows.append(row)
            study.save(out/'progress.json',dict(completed_cells=len(rows),expected_cells=2*len(cases),rows=rows))
        case.close_sql()
    summary=dict(status='pass',source_id=sid,rows=rows,overall=aggregate(rows),
        groups={f'{key}_b{budget}':aggregate([r for r in rows if r['key']==key and r['budget']==budget])
            for key in ('l1','l2') for budget in (125,500)
            if any(r['key']==key and r['budget']==budget for r in rows)})
    study.save(out/'summary.json',summary)
    print(json.dumps(summary['overall']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--pilot',action='store_true')
    main(p.parse_args())
