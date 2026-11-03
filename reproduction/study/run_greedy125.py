import os
"""Complete practical-greedy 125/side column for the constructed reference table."""
import json
from pathlib import Path
import sys
import time

from run_study import (HERE, ROOT, common, cases_for, greedy, sha, save, read,
    source_paths, load_pinned_reference, strict_check_record, SEEDS)


def run():
    cases=cases_for('constructed'); out=Path(os.environ.get('CERTIS_OUTPUT', HERE/'greedy125_v1'))
    binding=dict(sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()+[Path(__file__)]},
        targets={c.case_id:c.target_version for c in cases},budget=125,seeds=SEEDS,
        scope='Exploratory practical-greedy table completion; old records unchanged')
    import hashlib
    sid=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    manifest=dict(source_id=sid,**binding)
    if (out/'manifest.json').exists(): assert json.loads((out/'manifest.json').read_text())==json.loads(json.dumps(manifest))
    else: save(out/'manifest.json',manifest)
    rows=[]
    for c in cases:
        reference=None
        for seed in SEEDS:
            key=f'{c.case_id}__joint_greedy__s{seed}__b125'; path=out/'raw'/(key+'.json.gz')
            if path.exists(): record=read(path); assert record['source_id']==sid
            else:
                cpu,wall=time.process_time(),time.perf_counter(); r=greedy(c,seed,125)
                record=dict(key=key,case_id=c.case_id,method='joint_greedy',seed=seed,budget=125,
                    source_id=sid,status='pass',result=r,metadata=c.metadata,
                    target_version=c.target_version,oracle_version=c.oracle_version,
                    cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)
                save(path,record,True)
            if reference is None: reference=load_pinned_reference(c)
            record['validation']=strict_check_record(record,c,record_path=path,expected_key=key,
                expected_method='joint_greedy',expected_seed=seed,expected_budget=125,
                expected_source_id=sid,reference=reference,canonical_execution_method='count_allocation_feedback_audit_degree')
            save(path,record,True); r=record['result']
            rows.append(dict(case_id=c.case_id,seed=seed,budget=125,method='joint_greedy',
                requests=r['summary']['oracle_requests'],loss=r['output']['loss'],
                raw_sha256=sha(path),timings=r['timings']))
            save(out/'progress.json',dict(completed=len(rows),expected=48,rows=rows))
            print(json.dumps(dict(completed=len(rows),**rows[-1])),flush=True)
        c.close_sql()
    save(out/'summary.json',dict(source_id=sid,verified=len(rows),expected=48,rows=rows))


if __name__=='__main__': run()
