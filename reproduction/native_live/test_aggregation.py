"""Check that pooled and query-macro latency comparisons stay distinct."""
import math
from analyze_native_live import summarize, METHODS, SEEDS

rows=[]
for case,key,exact,joint in (('large','l1',100.,90.),('small','l2',10.,12.)):
    for method in METHODS:
        value=joint if method=='joint' else exact
        for seed in SEEDS:
            rows.append(dict(case_id=case,key=key,method=method,seed=seed,
                complete_seconds=value,requests=int(value),tokens_in=2*int(value),tokens_out=int(value),
                inference_seconds=value*.9,local_seconds=value*.1,batches=1))
result=summarize(rows)
comparison=next(r for r in result['comparisons'] if r['key']=='all' and r['baseline']=='batched_right')
assert math.isclose(comparison['pooled_time_saving'],1-102/110)
assert math.isclose(comparison['macro_time_saving'],-.05)
assert comparison['queries']==2 and comparison['matched_runs']==4
assert comparison['time_query_wins']==1 and comparison['time_query_losses']==1
assert math.isclose(comparison['worst_query_time_saving'],-.2)
assert comparison['matched_run_time_wins']==2 and comparison['matched_run_time_losses']==2
assert len(comparison['matched_run_savings'])==4
print('pass: pooled saving 7.27%, query macro -5%; repetitions do not become workloads')
