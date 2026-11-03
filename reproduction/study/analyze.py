"""Recompute paired statistics and complete-cost boundaries from verified traces."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VISUAL=ROOT/'reproduction/figures'


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def wtl(a,b): return [sum(x<y for x,y in zip(a,b)),sum(x==y for x,y in zip(a,b)),sum(x>y for x,y in zip(a,b))]
def saving(a,b): return 100*(1-sum(a)/sum(b))


def bound(delta_local,delta_batches):
    if delta_batches<0:
        return dict(condition='tau > threshold',seconds=max(0.,delta_local/-delta_batches))
    if delta_batches==0:
        return dict(condition='all tau' if delta_local<0 else 'never',seconds=None)
    if delta_local<0:
        return dict(condition='tau < threshold',seconds=-delta_local/delta_batches)
    return dict(condition='never',seconds=None)


def local_cost(row):
    # All policies share the same metadata/SQL setup; its common cost cancels.
    q=row['sql_timings']
    return row['timings']['algorithm_seconds']+q['fact_ingest_seconds']+q['returned_materialization_seconds']


def main():
    constructed=load(HERE/'constructed_v1/summary.json')
    native_uncached=load(HERE/'native_v1/summary.json')
    native=native_uncached
    assert constructed['expected']==constructed['verified']==192
    assert native['expected']==native['verified']==168
    cached_path=HERE/'native_cached_v1/summary.json'
    cache_summary=None
    if cached_path.exists():
        cache_summary=load(cached_path)
        assert cache_summary['expected']==cache_summary['verified']==72
        assert cache_summary['all_ledgers_masks_events_identical']
        native=dict(native_uncached)
        native['rows']=[r for r in native_uncached['rows'] if r['method']!='joint_greedy']+cache_summary['rows']
    old=load(HERE/'baseline/plot_data.json')
    original={ (x['case_id'],x['seed']):x for x in old['matched_cells'] }
    split_rows=constructed['rows']; by_method={}
    fixed_v2_path=HERE/'fixed_continuous_v2/summary.json'
    fixed_v2=None
    if fixed_v2_path.exists():
        fixed_v2=load(fixed_v2_path)
        assert fixed_v2['expected']==fixed_v2['verified']==48
        split_rows=[r for r in split_rows if r['method']!='fixed_continuous']+fixed_v2['rows']
    for method in sorted({r['method'] for r in split_rows}):
        a=[r for r in split_rows if r['method']==method]
        by_method[method]=dict(cells=len(a),requests=sum(x['requests'] for x in a),
            max_loss=max(x['loss'] for x in a),coverage_failures=sum(not x['coverage'] for x in a),
            conflict_freezes=sum(x['conflict'] for x in a))
    adaptive={(r['case_id'],r['seed']):r for r in split_rows if r['method']=='adaptive_continuous'}
    target=[adaptive[k]['requests'] for k in original]
    for field,name in [('separate_requests','global_separate_greedy'),('greedy_requests','joint_greedy'),('exact_requests','joint_dp')]:
        vals=[original[k][field] for k in original]
        by_method[name]=dict(cells=48,requests=sum(vals),saving_vs_adaptive_continuous_percent=saving(vals,target),
            wtl_vs_adaptive_continuous=wtl(vals,target),
            per_cell_saving_min_median_max=[min(100*(1-a/b) for a,b in zip(vals,target)),
                median(100*(1-a/b) for a,b in zip(vals,target)),max(100*(1-a/b) for a,b in zip(vals,target))])
    old_fixed=sum(x['marginal_requests'] for x in original.values())
    by_method['historical_fixed_float_grid']=dict(requests=old_fixed,
        note='New fixed-grid control uses rational thresholds and a feasible nonzero split for initial zero-gain ties; the 34-request pooled difference from historical floating/default-tie semantics is not treated as a pure split-update effect.')
    if fixed_v2:
        by_method['fixed_continuous_attained_v1']=dict(requests=sum(r['requests'] for r in constructed['rows'] if r['method']=='fixed_continuous'),
            note='Superseded comparator froze attained recalls and left rounding slack unused; final fixed control spends the full product budget.')
    archive=ROOT/'reproduction/runtime/greedy_count_profile/full'
    archive_hashes=load(archive/'paired_analysis.json')['raw_bindings']
    matched_archive_audits=0
    for cid,seed in original:
        key=f'{cid}__greedy_count_profile_feedback_audit_degree__s{seed}__b500'
        path=archive/'raw'/(key+'.json.gz'); assert sha(path)==archive_hashes[key]
        old_record=json.loads(gzip.decompress(path.read_bytes()))
        new_key=f'{cid}__adaptive_continuous__s{seed}__b500'
        new_record=json.loads(gzip.decompress((HERE/'constructed_v1/raw'/(new_key+'.json.gz')).read_bytes()))
        a,b=old_record['result'],new_record['result']
        assert [x for x in a['oracle_ledger'] if x['phase']=='audit']==[x for x in b['oracle_ledger'] if x['phase']=='audit']
        assert a['regions']==b['regions']
        matched_archive_audits+=1
    paired=defaultdict(dict)
    exact={}
    for r in native['rows']:
        if r['method'].startswith('batched'): exact[(r['case_id'],r['method'])]=r
        else: paired[(r['case_id'],r['seed'],r['budget'])][r['method']]=r
    cost_rows=[]
    for (cid,seed,budget),methods in paired.items():
        g=methods['joint_greedy']; a=methods['adaptive_continuous']
        for reference in ('batched_left','batched_right','adaptive_continuous'):
            e=a if reference=='adaptive_continuous' else exact[(cid,reference)]
            dc=local_cost(g)-local_cost(e); db=g['batches']-e['batches']
            cost_rows.append(dict(case_id=cid,seed=seed,budget=budget,reference=reference,
                greedy_requests=g['requests'],reference_requests=e['requests'],
                saving_percent=100*(1-g['requests']/e['requests']),delta_batches=db,delta_local_seconds=dc,
                boundary=bound(dc,db),greedy_tokens_in=g['tokens_in'],reference_tokens_in=e['tokens_in'],
                greedy_tokens_out=g['tokens_out'],reference_tokens_out=e['tokens_out']))
    groups=[]
    for variant in ('subreddit','link_id'):
        for budget in (125,500):
            a=[x for x in cost_rows if x['case_id'].endswith('__'+variant) and x['budget']==budget and x['reference']=='batched_right']
            compare=[x for x in cost_rows if x['case_id'].endswith('__'+variant) and x['budget']==budget and x['reference']=='adaptive_continuous']
            ga=[x['greedy_requests'] for x in a]; er=[x['reference_requests'] for x in a]
            dc=sum(x['delta_local_seconds'] for x in a)/len(a); db=sum(x['delta_batches'] for x in a)/len(a)
            groups.append(dict(variant=variant,budget=budget,cells=len(a),
                mean_greedy_requests=sum(ga)/len(a),saving_vs_right_percent=saving(ga,er),wtl_vs_right=wtl(ga,er),
                saving_vs_adaptive_percent=saving([x['greedy_requests'] for x in compare],[x['reference_requests'] for x in compare]),
                mean_delta_batches=db,mean_delta_local_seconds=dc,pooled_batch_latency_boundary=bound(dc,db),
                max_observed_loss=max(r['loss'] for r in native['rows'] if r['method']=='joint_greedy' and r['case_id'].endswith('__'+variant) and r['budget']==budget)))
    validation={}
    for name,summary in [('constructed_v1',constructed),('native_v1',native_uncached)]:
        path=HERE/name; records=[]
        for row in summary['rows']:
            key=f"{row['case_id']}__{row['method']}__s{row['seed']}__b{row['budget']}"
            p=path/'raw'/(key+'.json.gz'); assert sha(p)==row['raw_sha256']
            record=json.loads(gzip.decompress(p.read_bytes()))
            assert record['validation']['strict_validation'] and record['validation']['retained_completion_verified']
            records.append(record)
        # Same ordered original audit, not just a shared seed or budget label.
        audit_groups=defaultdict(list)
        for r in records:
            if r['budget']:
                audit=[x for x in r['result']['oracle_ledger'] if x['phase']=='audit']
                audit_groups[(r['case_id'],r['seed'],r['budget'])].append((audit,r['result']['regions']))
        assert all(all(v==vs[0] for v in vs) for vs in audit_groups.values())
        validation[name]=dict(records=len(records),all_raw_hashes_match=True,all_strict_checks_passed=True,
            paired_ordered_audits_identical=True,unique_audit_events=len(audit_groups),
            observed_loss_violations=sum(r['result']['output']['loss']>.1 for r in records))
    validation['historical_greedy_comparison']=dict(raw_hashes_verified=True,matched_ordered_audits=matched_archive_audits)
    if fixed_v2:
        validation['full_budget_fixed_split']=dict(records=48,earlier_v1_preserved=True,
            final_four_arm_matrix='144 unchanged v1 records plus 48 corrected fixed-continuous records')
    for stage,summary,method in [('fixed_continuous_v2',fixed_v2,'fixed_continuous'),
                                  ('native_cached_v1',cache_summary,'joint_greedy_cached')]:
        if summary is None: continue
        for row in summary['rows']:
            key=f"{row['case_id']}__{method}__s{row['seed']}__b{row['budget']}"
            path=HERE/stage/'raw'/(key+'.json.gz')
            assert sha(path)==row['raw_sha256']
            record=json.loads(gzip.decompress(path.read_bytes()))
            assert record['validation']['strict_validation'] and record['validation']['retained_completion_verified']
            if stage=='native_cached_v1': assert record['identical_uncached_execution']
    report=dict(constructed=by_method,native_groups=groups,native_cost_rows=cost_rows,validation=validation,
        source_summaries={p.relative_to(ROOT).as_posix():sha(p) for p in [HERE/'constructed_v1/summary.json',HERE/'native_v1/summary.json',HERE/'baseline/plot_data.json']},
        analysis_source_sha256=sha(__file__),scope='Exploratory fixed targets; seeds and reused exact controls are not independent statistical replicates.')
    if cache_summary:
        old_by_cell={(r['case_id'],r['seed'],r['budget']):r for r in native_uncached['rows'] if r['method']=='joint_greedy'}
        a=cache_summary['rows']; b=[old_by_cell[(r['case_id'],r['seed'],r['budget'])] for r in a]
        validation['cached_greedy']=dict(records=72,ledgers_masks_events_identical=True)
        report['cache_implementation']=dict(cells=72,
            uncached_median_algorithm_seconds=median(r['timings']['algorithm_seconds'] for r in b),
            cached_median_algorithm_seconds=median(r['timings']['algorithm_seconds'] for r in a),
            median_paired_algorithm_time_reduction_percent=median(100*(1-x['timings']['algorithm_seconds']/y['timings']['algorithm_seconds']) for x,y in zip(a,b)),
            scope='Same numerical policy, query-local pure-kernel memoization; every charged identity and allocation event preserved.')
        report['source_summaries'][str(cached_path.relative_to(ROOT))]=sha(cached_path)
    if fixed_v2:
        report['source_summaries'][str(fixed_v2_path.relative_to(ROOT))]=sha(fixed_v2_path)
    live=HERE/'local_live_v1/summary.json'
    if live.exists():
        report['local_live']=load(live)
        cid='goemotions_native__sent_positive__sent_negative__subreddit'
        crows={m:exact[(cid,m)] if m.startswith('batched') else paired[(cid,101,500)][m]
               for m in ('batched_right','adaptive_continuous','joint_greedy')}
        for m,values in report['local_live']['methods'].items():
            values['local_overhead_seconds']=local_cost(crows[m])
            values['trace_total_with_local_overhead_seconds']=values['measured_inference_seconds']+local_cost(crows[m])
    aux=load(HERE/'greedy125_v1/summary.json')
    assert aux['expected']==aux['verified']==48
    for row in aux['rows']:
        key=f"{row['case_id']}__joint_greedy__s{row['seed']}__b125"
        path=HERE/'greedy125_v1/raw'/(key+'.json.gz')
        assert sha(path)==row['raw_sha256']
        assert json.loads(gzip.decompress(path.read_bytes()))['validation']['strict_validation']
    validation['constructed_greedy125']=dict(records=48,all_raw_hashes_and_strict_checks_pass=True)
    report['source_summaries'][(HERE/'greedy125_v1/summary.json').relative_to(ROOT).as_posix()]=sha(HERE/'greedy125_v1/summary.json')
    save(HERE/'analysis.json',report)
    # Preserve every old plotted cell; add the new, stronger comparator alongside it.
    data=load(VISUAL/'plot_data.json')
    for x in data['matched_cells']:
        x['adaptive_per_side_requests']=adaptive[(x['case_id'],x['seed'])]['requests']
    data['strengthening']=dict(constructed=by_method,native_rows=native['rows'],native_groups=groups,
        analysis_sha256=sha(HERE/'analysis.json'),analysis_path='reproduction/study/analysis.json')
    historical_greedy=load(archive/'summary.json')['rows']
    new_greedy={('constructed',125):aux['rows'],
                ('constructed',500):[r for r in historical_greedy if r['family']=='constructed'],
                ('native',125):[r for r in historical_greedy if r['family']=='native' and r['budget']==125],
                ('native',500):[r for r in historical_greedy if r['family']=='native' and r['budget']==500]}
    if 'historical_dp_table_groups' not in data:
        data['historical_dp_table_groups']=data['table_groups']
    import copy
    tables=copy.deepcopy(data['historical_dp_table_groups'])
    source=ROOT/'reproduction/runtime/external_study/results/reference'
    exact_hashes={}; exact_count_cache={}
    for group in tables:
        family=group['cohort']
        for row in group['rows']:
            comparisons=[]
            for budget in (125,500):
                newvals=[]; refs=[]
                for cell in new_greedy[(family,budget)]:
                    cid=cell['case_id']; name=f"{cid}__{row['method']}__s0__b0.json.gz"
                    folders=[source/'raw',source/'extra_exact_controls/raw'] if family=='constructed' else [ROOT/'reproduction/runtime/native_discussion/execution/raw']
                    paths=[p/name for p in folders if (p/name).exists()]
                    assert len(paths)==1,(name,paths)
                    p=paths[0]; exact_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
                    if str(p) not in exact_count_cache:
                        raw=json.loads(gzip.decompress(p.read_bytes()))['result']
                        exact_count_cache[str(p)]=raw.get('summary',raw)['oracle_requests']
                    e=exact_count_cache[str(p)]
                    refs.append(e); newvals.append(cell['requests'])
                wins,ties,losses=wtl(newvals,refs)
                comparisons.append(dict(pairs=len(refs),new_total=sum(newvals),reference_total=sum(refs),
                    pooled_saving_fraction=1-sum(newvals)/sum(refs),wins=wins,ties=ties,losses=losses,
                    min_case_saving_fraction=min(1-a/b for a,b in zip(newvals,refs)),
                    max_case_saving_fraction=max(1-a/b for a,b in zip(newvals,refs))))
            row['comparisons']=comparisons
    data['table_groups']=tables
    save(HERE/'greedy_table_exact_sources.json',exact_hashes)
    save(VISUAL/'plot_data.json',data)
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Natural contrast tasks: six predicate pairs, three seeds per row. Saving uses batched exact right-first. $\Delta B$ and $\Delta C$ are mean extra batches and local seconds for greedy CERTIS.}',
        r'\label{tab:cost_boundary}',r'\begin{tabularx}{\linewidth}{Xrrrrr}',r'\toprule',
        r'\textbf{Keys} & \textbf{Audit} & \textbf{Saving} & $\Delta B$ & $\Delta C$ & $\tau^\ast$ \\',r'\midrule']
    for g in groups:
        keys='Subreddit' if g['variant']=='subreddit' else 'Thread'
        tau=g['pooled_batch_latency_boundary']['seconds']
        tau_text='---' if tau is None else f'{tau:.2f}'
        lines.append(f"{keys} & {g['budget']} & {g['saving_vs_right_percent']:.2f}"+r'\%'+f" & {g['mean_delta_batches']:+.2f} & {g['mean_delta_local_seconds']:+.2f} & {tau_text}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabularx}',
        r'\par\smallskip\raggedright Audit is per side. $\tau^\ast$ is the pooled break-even seconds per batch under a common constant batch latency; --- denotes no positive-latency break-even. Local time includes audit, planning, bookkeeping, fact ingest and returned SQL; reference verification is excluded.',r'\end{table}']
    (HERE/'cost_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(constructed=by_method,native_groups=groups,validation=validation),ensure_ascii=True,indent=2))


if __name__=='__main__': main()
