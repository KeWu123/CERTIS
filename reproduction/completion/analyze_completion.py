"""Recheck raw live acquisitions and aggregate all completion-study results."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import json
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import live_closed_loop as live
import pandas as pd


def read(path):
    if path.name.endswith('.gz'):return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(source,destination):
    counts=source/'counts'
    calibration=read(counts/'calibration.json')
    assert calibration['events']==len(calibration['rows'])==1200
    assert len({(r['case_id'],r['budget'],r['seed']) for r in calibration['rows']})==1200
    assert all(x==50 for x in Counter((r['case_id'],r['budget']) for r in calibration['rows']).values())
    geometry=read(counts/'mechanism_optima.json')['rows']
    assert len(geometry)==180
    assert all(r['joint_optimal_discard']>=r['separate_optimal_discard'] for r in geometry)
    canonical=[r for r in geometry if r['seed']==101]
    scale=read(counts/'practical_scaling.json')['rows']
    assert len(scale)==80
    scaling=[]
    for n,k,law in sorted({(r['n'],r['k'],r['law']) for r in scale}):
        selected=[r for r in scale if (r['n'],r['k'],r['law'])==(n,k,law)]
        timed=[r for r in selected if not r['trace_memory']]
        memory=[r for r in selected if r['trace_memory']]
        assert len(timed)==3 and len(memory)==1
        assert all(r['status']=='pass' for r in selected)
        scaling.append(dict(n=n,k=k,law=law,
            median_seconds=statistics.median(r['total_seconds'] for r in timed),
            minimum_seconds=min(r['total_seconds'] for r in timed),
            maximum_seconds=max(r['total_seconds'] for r in timed),
            traced_peak_bytes=memory[0]['traced_peak_bytes'],resident_peak_bytes=memory[0]['resident_peak_bytes'],
            cache_hits=memory[0]['cache']['hits'],cache_misses=memory[0]['cache']['misses']))
    closed=source/'closed_loop'
    manifest=read(closed/'manifest.json')
    reference=closed/'fixed_oracle.parquet'
    assert live.digest(reference)==read(closed/'reference_binding.json')['sha256']
    labels=pd.read_parquet(reference).set_index(['predicate_id','row_id']).label.to_dict()
    cases=live.fixed_subset([case for case in live.study.cases_for('native') if
        tuple(p.rsplit('/',1)[-1] for p in (case.predicate_left,case.predicate_right)) in live.PAIRS])
    cases={c.case_id:c for c in cases}
    records=sorted((closed/'raw').glob('*.json.gz'));assert len(records)==24
    rows=[]
    for path in records:
        record=read(path);row=record['row'];result=record['result'];case=cases[row['case_id']]
        assert record['binding']==manifest['binding']
        assert row['live_outputs_used_for_policy'] and row['all_live_labels_equal_fixed_reference']
        ledger=result['oracle_ledger'];batches=result['batches']
        assert row['requests']==len(ledger)==len({r['id'] for r in ledger})
        assert [j for b in batches for j in b['ids']]==[r['id'] for r in ledger]
        measured=[]
        for batch in batches:
            assert 0<len(batch['ids'])<=128
            for j,response in zip(batch['ids'],batch['responses']):
                predicate=case.predicate_left if case.side[j]==0 else case.predicate_right
                assert labels[predicate,case.row_ids[j]]==response['label']
                measured.append((j,response['label'],response['tokens_in'],response['tokens_out']))
        assert measured==[(r['id'],r['label'],r['tokens_in'],r['tokens_out']) for r in ledger]
        assert abs(row['inference_seconds']-sum(b['wall_seconds'] for b in batches))<1e-7
        assert row['tokens_in']==sum(r['tokens_in'] for r in ledger)
        assert row['tokens_out']==sum(r['tokens_out'] for r in ledger)
        facts={int(j):y for j,y in result['facts'].items()}
        assert facts=={r['id']:r['label'] for r in ledger}
        truth={j:int(labels[case.predicate_left if case.side[j]==0 else case.predicate_right,
                            case.row_ids[j]]) for j in range(case.n)}
        full=sum(sum(truth[j] for j in a)*sum(truth[j] for j in b) for a,b in zip(case.key_left,case.key_right))
        returned=sum(sum(facts.get(j)==1 for j in a)*sum(facts.get(j)==1 for j in b)
                     for a,b in zip(case.key_left,case.key_right))
        assert full==row['full_count'] and returned==row['returned_count']
        assert result['output']['bag_subset'] and result['output']['lost_count']==full-returned
        coverage=all(r['lower']<=sum(truth[j] for j in r['ids'])<=r['upper'] for r in result['regions'])
        assert coverage==row['coverage']
        assert not coverage or 10*(full-returned)<=full
        rows.append(row)
    assert Counter((r['case_id'],r['method']) for r in rows)==Counter(
        {(case,method):2 for case in cases for method in live.METHODS})
    methods={}
    for method in live.METHODS:
        selected=[r for r in rows if r['method']==method]
        methods[method]=dict(requests=sum(r['requests'] for r in selected),
            complete_seconds=sum(r['complete_seconds'] for r in selected),
            inference_seconds=sum(r['inference_seconds'] for r in selected),
            local_seconds=sum(r['local_seconds'] for r in selected))
    g=methods['joint_greedy_cached'];e=methods['batched_right'];s=methods['adaptive_continuous']
    case_rows=[]
    for case in cases:
        record=dict(case_id=case)
        for method in live.METHODS:
            selected=[r for r in rows if r['case_id']==case and r['method']==method]
            record[method]=dict(mean_seconds=statistics.mean(r['complete_seconds'] for r in selected),
                min_seconds=min(r['complete_seconds'] for r in selected),
                max_seconds=max(r['complete_seconds'] for r in selected),
                mean_requests=statistics.mean(r['requests'] for r in selected))
        case_rows.append(record)
    report=dict(status='pass',audit_events=1200,audit_failure_events=calibration['failure_events'],
        certificate_configurations=90,certificate_permutation_runs=180,
        strict_joint_optimal_gains=sum(r['exact_certificate_gain']>0 for r in canonical),
        maximum_extra_discard_fraction=max(r['exact_certificate_gain']/r['n'] for r in canonical),
        greedy_regressions=sum(r['extra_discard']<0 for r in geometry),scaling=scaling,
        live=dict(executions=24,queries=4,comments=2048,server_slots=1,methods=methods,cases=case_rows,
            pooled_latency_saving_vs_exact=1-g['complete_seconds']/e['complete_seconds'],
            pooled_latency_saving_vs_per_side=1-g['complete_seconds']/s['complete_seconds'],
            pooled_request_saving_vs_exact=1-g['requests']/e['requests'],
            coverage_failure_runs=sum(not r['coverage'] for r in rows),max_loss=max(r['loss'] for r in rows),
            paid_model_labels_rechecked=True),
        scopes='Fixed-oracle live quality uses Qwen, independently of archived 70B quality. '
            'Timing ranges include different audit seeds. Calibration events and count permutations are not independent query workloads.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('scaling','live')}))
    print(json.dumps(report['live'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'results')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/completion_analysis.json')
    args=p.parse_args();analyze(args.source,args.output)
