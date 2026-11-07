"""Independently check native live ledgers, bag counts and complete costs.

This analyzer never calls the model or changes a policy record. Query-level
summaries use the four declared, related queries; audit repetitions are not
counted as new workloads. Run after all 32 policy executions finish.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
RELEASE=WORKSPACE/'artifacts/CERTIS_release_20261104'
SOURCE=RELEASE/'reproduction/runtime/candidate_systems/external/semviews'
METHODS=('batched_right','adaptive_continuous','separate','joint')
PREDICATES=('agent','person','athlete')
SEEDS=(101,202)


def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(a,b)


def canonical_prompt(text,predicate,questions):
    return ('<|im_start|>system\nClassify the quoted article. Answer exactly true or false.<|im_end|>\n'
        '<|im_start|>user\nStatement: '+questions[predicate]+'.\nArticle: '+
        json.dumps(text,ensure_ascii=False)+'\nIs the statement true?\nAnswer:<|im_end|>\n'
        '<|im_start|>assistant\n')


def summarize(rows):
    cases=sorted({r['case_id'] for r in rows})
    query_rows=[]
    for case in cases:
        group=[r for r in rows if r['case_id']==case]
        item=dict(case_id=case,key=group[0]['key'],methods={})
        for method in METHODS:
            selected=[r for r in group if r['method']==method]
            assert {r['seed'] for r in selected}==set(SEEDS)
            item['methods'][method]=dict(
                mean_seconds=statistics.mean(r['complete_seconds'] for r in selected),
                min_seconds=min(r['complete_seconds'] for r in selected),
                max_seconds=max(r['complete_seconds'] for r in selected),
                mean_requests=statistics.mean(r['requests'] for r in selected),
                mean_inference_seconds=statistics.mean(r['inference_seconds'] for r in selected),
                mean_local_seconds=statistics.mean(r['local_seconds'] for r in selected),
                mean_tokens_in=statistics.mean(r['tokens_in'] for r in selected),
                mean_tokens_out=statistics.mean(r['tokens_out'] for r in selected),
                runs=selected)
        query_rows.append(item)
    comparisons=[]
    for key in ('all','l1','l2'):
        qs=[r for r in query_rows if key=='all' or r['key']==key]
        rs=[r for r in rows if key=='all' or r['key']==key]
        for baseline in ('batched_right','adaptive_continuous','separate'):
            target=[r for r in rs if r['method']=='joint']
            reference=[r for r in rs if r['method']==baseline]
            lookup={(r['case_id'],r['method'],r['seed']):r for r in rs}
            matched=[]
            for query in qs:
                for seed in SEEDS:
                    a=lookup[(query['case_id'],'joint',seed)]
                    b=lookup[(query['case_id'],baseline,seed)]
                    matched.append(dict(case_id=query['case_id'],seed=seed,
                        time_saving=1-a['complete_seconds']/b['complete_seconds'],
                        request_saving=1-a['requests']/b['requests']))
            rates=[1-q['methods']['joint']['mean_seconds']/q['methods'][baseline]['mean_seconds'] for q in qs]
            count_rates=[1-q['methods']['joint']['mean_requests']/q['methods'][baseline]['mean_requests'] for q in qs]
            comparisons.append(dict(key=key,baseline=baseline,queries=len(qs),matched_runs=len(target),
                pooled_time_saving=1-sum(r['complete_seconds'] for r in target)/sum(r['complete_seconds'] for r in reference),
                pooled_request_saving=1-sum(r['requests'] for r in target)/sum(r['requests'] for r in reference),
                pooled_input_token_saving=1-sum(r['tokens_in'] for r in target)/sum(r['tokens_in'] for r in reference),
                macro_time_saving=statistics.mean(rates),median_query_time_saving=statistics.median(rates),
                worst_query_time_saving=min(rates),macro_request_saving=statistics.mean(count_rates),
                time_query_wins=sum(r>0 for r in rates),time_query_ties=sum(r==0 for r in rates),
                time_query_losses=sum(r<0 for r in rates),
                matched_run_min_time_saving=min(r['time_saving'] for r in matched),
                matched_run_max_time_saving=max(r['time_saving'] for r in matched),
                matched_run_time_wins=sum(r['time_saving']>0 for r in matched),
                matched_run_time_ties=sum(r['time_saving']==0 for r in matched),
                matched_run_time_losses=sum(r['time_saving']<0 for r in matched),
                matched_run_savings=matched))
    totals={method:{field:sum(r[field] for r in rows if r['method']==method) for field in
        ('requests','tokens_in','tokens_out','inference_seconds','local_seconds','complete_seconds','batches')}
        for method in METHODS}
    return dict(queries=query_rows,comparisons=comparisons,totals=totals)


def analyze(source,output,allow_partial=False):
    manifest=read(source/'manifest.json');binding=manifest['binding']
    info={k:v for k,v in manifest.items() if k!='binding'}
    assert jsha(info)==binding
    assert manifest['rows_per_side']==5000 and manifest['audit_budget']==500
    assert tuple(manifest['methods'])==METHODS and tuple(manifest['seeds'])==SEEDS
    assert not manifest['original_70b_labels_used_for_policy'] and manifest['slots']==1
    assert not manifest['continuous_batching'] and not manifest['decoding']['cache_prompt']
    for relative,expected in manifest['sources'].items():
        original=WORKSPACE/relative.replace('\\','/')
        frozen=source/'source_versions'/(expected+Path(relative).suffix)
        assert frozen.is_file() and sha(frozen)==expected,relative
        assert original.is_file() and sha(original)==expected,relative
    assert not (source/'oracle_disagreement.json').exists()
    reference_binding=read(source/'reference_binding.json')
    reference_path=source/reference_binding['parquet']
    assert sha(reference_path)==reference_binding['sha256']
    reference=pd.read_parquet(reference_path)
    assert len(reference)==15000 and not reference.duplicated(['row_id','predicate_id']).any()
    assert set(reference.predicate_id)=={'dbpedia/'+p for p in PREDICATES}
    labels={(str(r.row_id),str(r.predicate_id)):int(r.label) for r in reference.itertuples()}
    assert set(labels.values())<={0,1}
    dataset=SOURCE/'data/datasets/dbpedia.parquet'
    assert sha(dataset)==manifest['data_sha256']
    frame=pd.read_parquet(dataset).sort_values('row_id').reset_index(drop=True)
    roster=list(map(str,frame.row_id))
    assert roster==manifest['roster'] and len(roster)==5000 and len(set(roster))==5000
    texts=list(map(str,frame.text))
    prompts=[canonical_prompt(text,p,manifest['predicates']) for p in PREDICATES for text in texts]
    assert jsha(prompts)==manifest['canonical_prompt_sha256']
    assert [jsha(p) for p in prompts]==manifest['per_prompt_sha256']
    repeatability=read(source/'repeatability.json')
    assert repeatability['binding']==binding
    assert {r['cap'] for r in repeatability['checks']}=={1,8,128}
    assert all(not r['mismatches'] for r in repeatability['checks'])
    token_check=read(source/'prompt_tokens.json')
    assert token_check['max_input_tokens']+token_check['output_allowance']<=token_check['context']
    assert len(token_check['counts'])==15000
    expected={(case,method,seed) for case in manifest['cases'] for method in METHODS for seed in SEEDS}
    audits={};regions={};audit_coverage={};rows=[];raw_bindings={};records_by_key={};record_file_times={}
    records=list((source/'raw').glob('*.json.gz'))
    assert len(expected)==32 and 0<len(records)<=32
    if not allow_partial:assert len(records)==32
    for path in sorted(records):
        record=read(path);row=record['row'];result=record['result']
        assert record['binding']==binding and record['status']=='pass'
        key=(row['case_id'],row['method'],row['seed']);assert key in expected;expected.remove(key)
        assert record['repetition']==SEEDS.index(row['seed'])
        records_by_key[key]=row
        record_file_times[path.name]=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()
        _,left,right,column=row['case_id'].split('__')
        assert column==row['key'] and column in ('l1','l2')
        predicates=['dbpedia/'+left]*5000+['dbpedia/'+right]*5000
        truth=np.array([labels[(rid,pred)] for rid,pred in zip(roster*2,predicates)],dtype=np.int8)
        identities=[p+'|'+rid for rid,p in zip(roster*2,predicates)]
        ledger=result['oracle_ledger'];facts={int(j):int(y) for j,y in result['facts'].items()}
        ids=[int(r['id']) for r in ledger]
        assert len(ids)==len(set(ids))==row['requests']==result['summary']['oracle_requests']
        assert set(ids)==set(facts) and all(0<=j<10000 for j in ids)
        assert result['summary']['initial_cache_count']==0
        assert not result['helper_ledger'] and row['helper_requests']==0
        for paid in ledger:
            j=int(paid['id']);assert paid['identity']==identities[j]
            assert int(paid['label'])==facts[j]==int(truth[j])
            assert paid['phase'] in ('audit','execution')
        batches=result['batches'];assert len(batches)==row['batches']
        assert [int(j) for b in batches for j in b['ids']]==ids
        cursor=0
        for batch in batches:
            assert batch['size']==len(batch['ids'])==len(batch['responses'])<=128
            for index,(j,response) in enumerate(zip(batch['ids'],batch['responses'])):
                paid=ledger[cursor];cursor+=1
                assert paid['phase']==batch['phase'] and paid['id']==j
                assert response['label']==paid['label']
                assert response['tokens_in']==paid['tokens_in'] and response['tokens_out']==paid['tokens_out']
                raw=response['raw'];assert raw['content'].strip()==('true' if paid['label'] else 'false')
                assert not raw.get('truncated') and raw.get('timings',{}).get('cache_n',0)==0
                predicate=left if j<5000 else right
                assert batch['prompt_sha256'][index]==jsha(canonical_prompt(texts[j%5000],predicate,manifest['predicates']))
        for field in ('tokens_in','tokens_out'):
            assert sum(r[field] for r in ledger)==row[field]
        audit=[r['id'] for r in ledger if r['phase']=='audit']
        assert len(audit)==(0 if row['method']=='batched_right' else 1000)
        assert ids[:len(audit)]==audit
        groups=frame.groupby(column,sort=True).indices
        if row['method']!='batched_right':
            rng=np.random.default_rng(row['seed'])
            expected_audit=[int(j) for side in (0,1)
                for j in rng.choice(np.arange(side*5000,(side+1)*5000),size=500,replace=False)]
            assert audit==expected_audit
            sample_set=set(audit)
            domains={tuple(int(j)+side*5000 for j in members)
                for members in groups.values() for side in (0,1)}
            assert {tuple(r['ids']) for r in result['regions']}==domains
            assert len(result['regions'])==len(domains)
            for region in result['regions']:
                domain=tuple(region['ids'])
                assert region['population']==len(domain)
                assert region['sample']==sum(j in sample_set for j in domain)
                assert region['positive_sample']==sum(facts[j] for j in domain if j in sample_set)
                assert Fraction(region['alpha_nominal'])==Fraction(1,20*len(domains))
                assert 0<Fraction(region['alpha_executed'])<=Fraction(region['alpha_nominal'])
            audit_key=(row['case_id'],row['seed'])
            if audit_key in audits:assert audit==audits[audit_key] and result['regions']==regions[audit_key]
            else:audits[audit_key]=audit;regions[audit_key]=result['regions']
        discarded=set(result['discarded']);assert not discarded.intersection(facts)
        initial=set(result['initial_retained_unknown'])
        assert {r['id'] for r in ledger if r['phase']=='execution'}<=initial
        full_count=returned_count=0
        for members in groups.values():
            left_ids=list(map(int,members));right_ids=[j+5000 for j in left_ids]
            full_count+=sum(int(truth[j]) for j in left_ids)*sum(int(truth[j]) for j in right_ids)
            returned_count+=sum(facts.get(j,0) for j in left_ids)*sum(facts.get(j,0) for j in right_ids)
            alive=lambda j:facts[j]==1 if j in facts else j not in discarded
            la,ra=sum(alive(j) for j in left_ids),sum(alive(j) for j in right_ids)
            assert not any(j not in facts and j not in discarded and ra for j in left_ids)
            assert not any(j not in facts and j not in discarded and la for j in right_ids)
        materialized=result['output']
        assert full_count==row['full_count']==materialized['full_count']
        assert returned_count==row['returned_count']==materialized['returned_count']
        assert full_count-returned_count==materialized['lost_count']
        close(row['loss'],(full_count-returned_count)/full_count if full_count else 0)
        assert materialized['bag_subset'] and materialized['cold_returned_first']
        assert materialized['sql_evaluation_order']=='returned-before-reference'
        coverage=all(r['lower']<=sum(int(truth[int(j)]) for j in r['ids'])<=r['upper'] for r in result['regions'])
        assert coverage==row['coverage']
        if row['method']!='batched_right':
            audit_key=(row['case_id'],row['seed'])
            if audit_key in audit_coverage:assert audit_coverage[audit_key]==coverage
            else:audit_coverage[audit_key]=coverage
        if coverage:assert 10*(full_count-returned_count)<=full_count
        if row['method']=='batched_right':assert full_count==returned_count
        inference=sum(b['wall_seconds'] for b in batches);close(inference,row['inference_seconds'])
        sql=sum(materialized[k] for k in ('sql_preprocessing_seconds','fact_ingest_seconds','returned_materialization_seconds'))
        close(sql,row['returned_sql_seconds'])
        raw_complete=result['timings']['algorithm_seconds']+sql
        close(raw_complete,row['complete_with_inline_monitor_seconds'])
        close(raw_complete-row['inline_oracle_validation_seconds'],row['complete_seconds'])
        close(row['complete_seconds']-inference,row['local_seconds'])
        close(row['outer_seconds']-row['complete_seconds'],row['excluded_verification_seconds'])
        assert row['local_seconds']>=-1e-6
        phase_wall={p:sum(b['wall_seconds'] for b in batches if b['phase']==p) for p in ('audit','execution')}
        for phase in phase_wall:close(phase_wall[phase],row['inference_by_phase'][phase])
        timing=result['timings']
        audit_local=timing.get('audit_seconds',0)-phase_wall['audit']
        projection=timing.get('projection_seconds',0)
        preproc=timing.get('preprocessing_seconds',0)
        residual=timing['algorithm_seconds']-inference-audit_local-projection-preproc-row['inline_oracle_validation_seconds']
        row=dict(row,cost_breakdown=dict(inference=inference,raw_audit_beyond_inference=audit_local,
            projection=projection,preprocessing=preproc,remaining_execution_less_all_inline_monitor=residual,
            sql_setup=materialized['sql_preprocessing_seconds'],fact_ingest=materialized['fact_ingest_seconds'],
            returned_materialization=materialized['returned_materialization_seconds']),
            cost_breakdown_scope='Raw audit-beyond-inference includes its inline validation monitor; '
                'the remaining execution residual subtracts the entire monitor total. '
                'Only their sum is a monitor-free local overhead; phase-specific monitor allocation was not recorded.',
            independent_counts_rechecked=True)
        close(sum(row['cost_breakdown'].values()),row['complete_seconds'])
        rows.append(row);raw_bindings[path.name]=sha(path)
    if not expected:
        summary=read(source/'summary.json');assert summary['verified']==32 and summary['fixed_reference_disagreements']==0
        assert summary['status']=='pass' and summary['binding']==binding and summary['expected']==32
        assert len(audit_coverage)==8
        declared_sequence=[]
        for repetition,seed in enumerate(SEEDS):
            for ci,case in enumerate(manifest['cases']):
                shift=(repetition+ci)%len(METHODS)
                declared_sequence.extend((case,method,seed) for method in METHODS[shift:]+METHODS[:shift])
        actual_sequence=[(r['case_id'],r['method'],r['seed']) for r in summary['rows']]
        assert actual_sequence==declared_sequence
        assert all(r==records_by_key[k] for r,k in zip(summary['rows'],actual_sequence))
    report=dict(status='incomplete' if expected else 'pass',validated_runs=len(rows),remaining_runs=len(expected),
        binding=binding,rows=rows,raw_sha256=raw_bindings,
        raw_file_last_write_utc=record_file_times,
        raw_file_time_scope='Filesystem last-write observations collected by the independent analyzer; '
            'these are record persistence times, not exact model-call start/end timestamps. '
            'Portable copies may have different filesystem times; raw bytes and hashes are the scientific binding.',
        execution_sequence_checked=not expected,
        all_paid_model_labels_rechecked=True,independent_factorized_bag_counts=True,
        returned_first_sql=True,reference_requests=len(reference),reference_cost=reference_binding,
        distinct_audit_events=len(audit_coverage),
        coverage_failure_audit_events=sum(not value for value in audit_coverage.values()),
        audit_event_coverage=[dict(case_id=key[0],seed=key[1],covered=value)
            for key,value in sorted(audit_coverage.items())],
        startup=read(source/'startup.json'),preflight=read(source/'preflight_summary.json'),
        reference_positive_counts={p:int(reference.loc[reference.predicate_id=='dbpedia/'+p,'label'].sum())
                                   for p in PREDICATES},
        coverage_failure_runs=sum(not r['coverage'] for r in rows),max_loss=max(r['loss'] for r in rows),
        conflict_runs=sum(r['conflict'] for r in rows),**({} if expected else summarize(rows)),
        scope='Four related native queries, two audit seeds each. Resident single-slot service, '
            'serial GPU inference inside logical batches; no general batching or significance claim. '
            'Reference/startup/warmup are separate; actual paid labels drive all policies. '
            'Audit-beyond-inference includes sampling, inversion and its bookkeeping; projection is not added twice.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','binding','coverage_failure_runs','max_loss','conflict_runs','totals','comparisons')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'outputs/full5000_v1')
    p.add_argument('--output',type=Path,default=HERE/'native_live_analysis.json')
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();analyze(args.source.resolve(),args.output.resolve(),args.allow_partial)
