import os
"""Full-budget fixed continuous split; separate correction of one comparator."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from run_study import (HERE,ROOT,cases_for,source_paths,sha,save,read,SEEDS,
    load_pinned_reference,strict_check_record)


spec=importlib.util.spec_from_file_location('_fixed_split_full_budget_harness',HERE/'run_study.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
Original=module.MarginalEnvelope


class FullBudget(Original):
    def plan(self,facts,old):
        result=super().plan(facts,old)
        if self.calls==1:
            for k,(left,right) in self.fixed.items():
                assert left>0
                self.fixed[k]=(left,Fraction(9,10)/left)
        return result


module.MarginalEnvelope=FullBudget


def main():
    cases=cases_for('constructed'); out=Path(os.environ.get('CERTIS_OUTPUT', HERE/'fixed_continuous_v2'))
    sources=source_paths()+[Path(__file__),HERE/'FIXED_SPLIT_ADDENDUM.md']
    binding=dict(sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        targets={c.case_id:c.target_version for c in cases},seeds=SEEDS,budget=500,
        fixed_right_required_recall='(9/10)/initial_left_attained_recall',
        old_results_preserved=True)
    sid=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    manifest=dict(source_id=sid,**binding)
    if (out/'manifest.json').exists(): assert json.loads((out/'manifest.json').read_text())==json.loads(json.dumps(manifest))
    else: save(out/'manifest.json',manifest)
    rows=[]
    for c in cases:
        reference=None
        for seed in SEEDS:
            method='fixed_continuous'; key=f'{c.case_id}__{method}__s{seed}__b500'; path=out/'raw'/(key+'.json.gz')
            if path.exists(): record=read(path); assert record['source_id']==sid
            else:
                cpu,wall=time.process_time(),time.perf_counter(); result=module.marginal(c,seed,500,method)
                record=dict(key=key,case_id=c.case_id,method=method,seed=seed,budget=500,result=result,
                    source_id=sid,status='pass',metadata=c.metadata,target_version=c.target_version,
                    oracle_version=c.oracle_version,policy_cpu_seconds=time.process_time()-cpu,
                    policy_wall_seconds=time.perf_counter()-wall)
                save(path,record,True)
            if reference is None: reference=load_pinned_reference(c)
            record['validation']=strict_check_record(record,c,record_path=path,expected_key=key,
                expected_method=method,expected_seed=seed,expected_budget=500,expected_source_id=sid,
                reference=reference,canonical_execution_method='count_allocation_feedback_audit_degree')
            save(path,record,True); r=record['result']; s=r['summary']
            rows.append(dict(case_id=c.case_id,method=method,seed=seed,budget=500,
                requests=s['oracle_requests'],loss=r['output']['loss'],
                coverage=record['validation']['joint_count_coverage_observed'],
                conflict=r['conflict_freeze'],raw_sha256=sha(path)))
            save(out/'progress.json',dict(completed=len(rows),expected=48,rows=rows))
            print(json.dumps(dict(completed=len(rows),**rows[-1])),flush=True)
        c.close_sql()
    save(out/'summary.json',dict(source_id=sid,verified=48,expected=48,rows=rows))


if __name__=='__main__': main()
