"""Validate and aggregate every native category-key query, including regressions."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import native_dbpedia as experiment


def analyze(source,output):
    manifest=json.loads((source/'manifest.json').read_text())
    cases={c.case_id:c for c in experiment.cases_for()}
    paths=sorted((source/'raw').glob('*.json.gz'));assert len(paths)==224
    rows=[]
    for path in paths:
        record=experiment.study.read(path);row=record['row'];result=record['result'];case=cases[row['case_id']]
        assert record['source_id']==manifest['source_id']
        truth=experiment.common.Oracle(case)._validation_labels()
        ledger=result['oracle_ledger'];facts={int(j):y for j,y in result['facts'].items()}
        assert len(ledger)==len({r['id'] for r in ledger})==row['requests']
        assert {r['id']:r['label'] for r in ledger}==facts
        assert all(int(truth[r['id']])==r['label'] for r in ledger)
        assert [j for b in result['batches'] for j in b['ids']]==[r['id'] for r in ledger]
        full=sum(sum(int(truth[j]) for j in a)*sum(int(truth[j]) for j in b)
                 for a,b in zip(case.key_left,case.key_right))
        returned=sum(sum(facts.get(j)==1 for j in a)*sum(facts.get(j)==1 for j in b)
                     for a,b in zip(case.key_left,case.key_right))
        assert result['output']['bag_subset'] and full==row['full_count'] and returned==row['returned_count']
        coverage=all(r['lower']<=sum(int(truth[j]) for j in r['ids'])<=r['upper'] for r in result['regions'])
        assert coverage==row['coverage'] and (not coverage or 10*(full-returned)<=full)
        for direction in ('in','out'):
            assert row['tokens_'+direction]==sum(r['tokens_'+direction] for r in ledger)
        rows.append(row)
    indexed={(r['case_id'],r['method'],r['seed'],r['budget']):r for r in rows}
    assert len(indexed)==224
    groups=[]
    for key in ('l1','l2'):
        for budget in (125,500):
            selected=[r for r in rows if r['key']==key and r['budget']==budget and r['method']=='joint_greedy_cached']
            assert len(selected)==24
            exact=[indexed[r['case_id'],'batched_right',0,0] for r in selected]
            side=[indexed[r['case_id'],'adaptive_continuous',r['seed'],budget] for r in selected]
            def contrast(reference):
                return dict(request_saving=1-sum(r['requests'] for r in selected)/sum(r['requests'] for r in reference),
                    wins=sum(r['requests']<e['requests'] for r,e in zip(selected,reference)),
                    ties=sum(r['requests']==e['requests'] for r,e in zip(selected,reference)),
                    losses=sum(r['requests']>e['requests'] for r,e in zip(selected,reference)))
            delta_c=sum(r['local_seconds']-e['local_seconds'] for r,e in zip(selected,exact))/24
            delta_b=sum(r['batches']-e['batches'] for r,e in zip(selected,exact))/24
            if delta_b<0:
                boundary_type='lower';threshold=max(0.,delta_c/(-delta_b))
            elif delta_b>0 and delta_c<0:
                boundary_type='upper';threshold=-delta_c/delta_b
            elif delta_b==0 and delta_c<0:
                boundary_type='all_positive';threshold=0.
            else:
                boundary_type='tie' if delta_b==0 and delta_c==0 else 'none';threshold=None
            groups.append(dict(key=key,budget=budget,cells=24,versus_exact=contrast(exact),
                versus_per_side=contrast(side),mean_extra_local_seconds=delta_c,mean_extra_batches=delta_b,
                break_even_seconds_per_batch=threshold,boundary_type=boundary_type,
                max_loss=max(r['loss'] for r in selected),coverage_failures=sum(not r['coverage'] for r in selected)))
    greedy=[r for r in rows if r['method']=='joint_greedy_cached']
    side=[indexed[r['case_id'],'adaptive_continuous',r['seed'],r['budget']] for r in greedy]
    exact=[indexed[r['case_id'],'batched_right',0,0] for r in greedy]
    left=[indexed[r['case_id'],'batched_left',0,0] for r in greedy]
    best=[min((a,b),key=lambda x:x['requests']) for a,b in zip(left,exact)]
    report=dict(status='pass',queries=16,executions=224,approximate_cells_per_method=96,groups=groups,
        pooled_saving_vs_exact=1-sum(r['requests'] for r in greedy)/sum(r['requests'] for r in exact),
        pooled_saving_vs_left=1-sum(r['requests'] for r in greedy)/sum(r['requests'] for r in left),
        pooled_saving_vs_retrospective_best_direction=1-sum(r['requests'] for r in greedy)/sum(r['requests'] for r in best),
        pooled_saving_vs_per_side=1-sum(r['requests'] for r in greedy)/sum(r['requests'] for r in side),
        wins_vs_exact=sum(r['requests']<e['requests'] for r,e in zip(greedy,exact)),
        ties_vs_exact=sum(r['requests']==e['requests'] for r,e in zip(greedy,exact)),
        losses_vs_exact=sum(r['requests']>e['requests'] for r,e in zip(greedy,exact)),
        max_loss=max(r['loss'] for r in rows),coverage_failure_records=sum(not r['coverage'] for r in rows),
        zero_answer_queries=sum(r['method']=='batched_right' and r['full_count']==0 for r in rows),
        raw_rows=rows,protocol_sha256=manifest['sources']['reproduction/completion/NATIVE_DBPEDIA_PROTOCOL.md'],
        scope='Known released category keys correlate with predicates. All eight pairs and both key levels retained. '
            'Oracle labels are charged fixed 70B labels; this is not a live latency benchmark.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='raw_rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'results/native_dbpedia')
    p.add_argument('--output',type=Path,default=experiment.ROOT/'outputs/native_dbpedia_analysis.json')
    args=p.parse_args();analyze(args.source,args.output)
