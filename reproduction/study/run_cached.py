import os
"""Validate identical ledgers/events for all 72 natural cached-greedy runs."""
import hashlib
import json
from pathlib import Path
import time

from run_study import (HERE,ROOT,cases_for,source_paths,sha,save,read,convert,
                      load_pinned_reference,strict_check_record,SEEDS)
from cached_greedy import execute


def normalize(x): return json.loads(json.dumps(x,default=convert))


def main():
    cases=cases_for('native'); out=Path(os.environ.get('CERTIS_OUTPUT', HERE/'native_cached_v1'))
    sources=source_paths()+[Path(__file__),HERE/'cached_greedy.py',HERE/'CACHE_PROTOCOL.md']
    binding=dict(sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        reference_manifest_sha256=sha(HERE/'native_v1/manifest.json'),
        reference_summary_sha256=sha(HERE/'native_v1/summary.json'),
        targets={c.case_id:c.target_version for c in cases},seeds=SEEDS,budgets=[125,500],
        cache_reset='per-query',max_cache_entries_per_kernel=65536)
    sid=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    manifest=dict(source_id=sid,**binding)
    if (out/'manifest.json').exists(): assert normalize(manifest)==json.loads((out/'manifest.json').read_text())
    else: save(out/'manifest.json',manifest)
    rows=[]
    for c in cases:
        reference=None
        for budget in (125,500):
            for seed in SEEDS:
                oldkey=f'{c.case_id}__joint_greedy__s{seed}__b{budget}'
                original=read(HERE/'native_v1/raw'/(oldkey+'.json.gz'))
                key=f'{c.case_id}__joint_greedy_cached__s{seed}__b{budget}'; path=out/'raw'/(key+'.json.gz')
                if path.exists(): record=read(path); assert record['source_id']==sid
                else:
                    cpu,wall=time.process_time(),time.perf_counter(); r=execute(c,seed,budget)
                    record=dict(key=key,case_id=c.case_id,method='joint_greedy_cached',seed=seed,budget=budget,
                        status='pass',source_id=sid,result=r,metadata=c.metadata,target_version=c.target_version,
                        oracle_version=c.oracle_version,policy_cpu_seconds=time.process_time()-cpu,
                        policy_wall_seconds=time.perf_counter()-wall)
                    save(path,record,True)
                r=record['result']; before=original['result']
                for field in ('oracle_ledger','facts','discarded','initial_retained_unknown','initial_discarded','events','regions','batches'):
                    assert normalize(r[field])==before[field],(key,field)
                if reference is None: reference=load_pinned_reference(c)
                record['validation']=strict_check_record(record,c,record_path=path,expected_key=key,
                    expected_method='joint_greedy_cached',expected_seed=seed,expected_budget=budget,
                    expected_source_id=sid,reference=reference,canonical_execution_method='count_allocation_feedback_audit_degree')
                record['identical_uncached_execution']=True; save(path,record,True)
                summary=r['summary']; output=r['output']
                row=dict(case_id=c.case_id,method='joint_greedy',seed=seed,budget=budget,
                    requests=summary['oracle_requests'],tokens_in=summary['oracle_tokens_in'],tokens_out=summary['oracle_tokens_out'],
                    phase_counts=summary['phase_counts'],batches=len(r['batches']),
                    cpu_seconds=record['policy_cpu_seconds'],wall_seconds=record['policy_wall_seconds'],
                    timings=r['timings'],sql_timings={k:output[k] for k in ('sql_preprocessing_seconds','fact_ingest_seconds',
                        'returned_materialization_seconds','reference_materialization_seconds','reference_label_loading_seconds','sql_validation_seconds')},
                    loss=output['loss'],full_count=output['full_count'],coverage=record['validation']['joint_count_coverage_observed'],
                    conflict=r['conflict_freeze'],raw_sha256=sha(path),uncached_wall_seconds=original['policy_wall_seconds'],
                    cache_info=r['cache_info'])
                rows.append(row); save(out/'progress.json',dict(completed=len(rows),expected=72,rows=rows))
                print(json.dumps(dict(completed=len(rows),case_id=c.case_id,seed=seed,budget=budget,
                    algorithm_seconds=r['timings']['algorithm_seconds'],identical=True)),flush=True)
        c.close_sql()
    save(out/'summary.json',dict(source_id=sid,expected=72,verified=72,all_ledgers_masks_events_identical=True,rows=rows))


if __name__=='__main__': main()
