"""Frozen-source exploratory count allocation; all structurally eligible cases."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,shutil,sys,time,traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from common import load_cases
from count_allocation_arms import eligible,run_count_allocation
from validate_results import check_record,labels_sql
from run_study import save

SID='db17c4543b8832d2acf2fc3db26099461a7055dc1ed64ad0867c0e5a47c8c22f'
FILES=('code/count_mask_planner.py','code/bilinear_certificate.py',
    'code/exact_hypergeom.py',
    'reproduction/runtime/external_study/count_allocation_arms.py',
    'reproduction/runtime/external_study/direct_arms.py',
    'reproduction/runtime/external_study/common.py',
    'reproduction/runtime/external_study/run_study.py',
    'reproduction/runtime/external_study/validate_results.py',
    'reproduction/runtime/external_study/run_count_allocation.py',
    'reproduction/runtime/external_study/COUNT_ALLOCATION_PROTOCOL.md')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=HERE/'results'/SID/'count_allocation'
    output.mkdir(parents=True,exist_ok=True)
    hashes={name:sha(ROOT/name) for name in FILES}
    source_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest_path=output/'source_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['source_hashes']==hashes,'Sources changed; retain old outputs in a separate study'
    else:
        manifest=dict(source_id=source_id,created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes)
        for name in FILES:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cases=load_cases();selected=[c for c in cases if eligible(c)]
    input_hashes={f'{c.dataset}:{name}':value for c in cases for name,value in c.metadata['source_hashes'].items()}
    input_hashes.update({c.dataset+':validated_array':c.metadata['array_sha256'] for c in cases})
    input_path=output/'input_manifest.json'
    if input_path.exists():assert json.loads(input_path.read_text(encoding='utf-8'))==input_hashes
    else:input_path.write_text(json.dumps(input_hashes,indent=2),encoding='utf-8')
    assert len(cases)==32 and len(selected)==16
    selection=dict(eligible=[c.case_id for c in selected],excluded=[c.case_id for c in cases if not eligible(c)],
                   reason='Metadata-only at-most-one-key incidence; no outcome selection')
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    checks=[];rows=[];failures=[]
    for case in selected:
        truth=None  # Read original-parquet truth only after the first policy.
        cells=[(s,b,True) for b in (125,500) for s in (101,202,303)]+[(101,0,False)]
        for seed,budget,audit in cells:
            method='count_allocation_feedback_'+('audit' if audit else 'point')+'_degree'
            key=f'{case.case_id}__{method}__s{seed}__b{budget}'
            path=output/'raw'/(key+'.json.gz')
            if path.exists():
                record=json.loads(gzip.decompress(path.read_bytes()))
                assert record['supplement_source_id']==source_id
            else:
                assert all(sha(ROOT/name)==value for name,value in hashes.items()),'Sources changed during execution'
                record=dict(key=key,case_id=case.case_id,dataset=case.dataset,variant=case.variant,
                    method=method,seed=seed,budget=budget,source_id=SID,supplement_source_id=source_id,
                    target_version=case.target_version,oracle_version=case.oracle_version,
                    metadata=case.metadata,status='pass',post_matrix_exploratory=True)
                started=time.perf_counter()
                try:record['result']=run_count_allocation(case,seed,budget,audit=audit)
                except Exception:
                    record['status']='failure';record['error']=traceback.format_exc()
                record['wall_seconds_including_materialization']=time.perf_counter()-started
                save(path,record)
            try:
                assert record['status']=='pass',record.get('error')
                if truth is None:truth=labels_sql(case)
                checks.append(check_record(record,case,truth))
                r=record['result'];summary=r['summary']
                # Original source/result identifiers are immutable comparators.
                if audit:
                    old_method='direct_feedback_audit_degree'
                    oldkey=f'{case.case_id}__{old_method}__s{seed}__b{budget}'
                    oldpath=HERE/'results'/SID/'raw'/(oldkey+'.json.gz')
                else:
                    old_method='direct_feedback_point_degree'
                    oldkey=f'{case.case_id}__{old_method}__s0__b0'
                    oldpath=HERE/'results'/SID/'supplement/raw'/(oldkey+'.json.gz')
                prior=json.loads(gzip.decompress(oldpath.read_bytes()))['result']
                assert prior['regions']==r['regions'],'Audit/evidence mismatch'
                old_mask=set(range(case.n))-set(prior['initial_retained_unknown'])
                old_mask-=set(a['id'] for a in prior['oracle_ledger'] if a['phase']=='audit')
                assert r['initial_certified_discard_count']>=len(old_mask),'All-mask theorem violated'
                rows.append(dict(case_id=case.case_id,seed=seed,budget=budget,audit=audit,
                    requests=summary['oracle_requests'],prior_requests=prior['summary']['oracle_requests'],
                    tokens=summary['oracle_tokens_in']+summary['oracle_tokens_out'],
                    prior_tokens=prior['summary']['oracle_tokens_in']+prior['summary']['oracle_tokens_out'],
                    initial_discard=r['initial_certified_discard_count'],prior_initial_discard=len(old_mask),
                    final_discard=r['final_certified_discard_count'],loss=r['output']['loss'],
                    cpu=r['timings']['algorithm_seconds'],conflict_freeze=r['conflict_freeze'],
                    comparator_sha256=sha(oldpath)))
            except Exception:failures.append(dict(key=key,error=traceback.format_exc()))
            print(json.dumps(dict(key=key,status=record['status'],done=len(rows),failures=len(failures))),flush=True)
        case.close_sql()
    assert len(rows)+len(failures)==112
    result=dict(status='pass' if not failures else 'failure',runs=len(rows),source_id=source_id,
                rows=rows,failures=failures,checks=checks)
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'result_manifest.json').write_text(json.dumps({str(p.relative_to(output)):sha(p) for p in sorted((output/'raw').glob('*.gz'))},indent=2),encoding='utf-8')
    if failures:raise SystemExit(1)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
