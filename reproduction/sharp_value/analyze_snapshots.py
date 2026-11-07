"""Descriptive complete-matrix diagnostics; no result-dependent cell selection."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import statistics

from sharp_wrapper import HERE, exact, mask_support, study


def one(snapshot,record):
    states=tuple((exact.SideCounts(**k['left']),exact.SideCounts(**k['right']))
        for k in snapshot['keys'])
    output=dict(case_id=record['case_id'],key=record['key'],budget=record['budget'],
        seed=record['seed'],trajectory=record['method'],index=snapshot['index'],
        paid=snapshot['paid'],same_evidence_discard_gap=snapshot['same_evidence_discard_gap'],
        alternatives=snapshot['alternatives'],keys=[])
    for index,(key,(a,b)) in enumerate(zip(snapshot['keys'],states)):
        row=dict(key)
        row['selected_masks']={}
        for kind,answer in snapshot['alternatives'].items():
            if answer is None:
                row['selected_masks'][kind]=None
                continue
            count=answer['counts'][index]
            hj=mask_support(((a,b),),(count,),Fraction(1,10))
            hs=mask_support(((a,b),),(count,),Fraction(1,10),True)
            assert hj<=hs
            row['selected_masks'][kind]=dict(counts=count,joint_support=hj,
                separate_support=hs,decoupling_gap=hs-hj)
        row['effective_interval_width_left']=min(a.upper,a.positive+a.unknown)-max(a.lower,a.positive)
        row['effective_interval_width_right']=min(b.upper,b.positive+b.unknown)-max(b.lower,b.positive)
        output['keys'].append(row)
    for kind,answer in snapshot['alternatives'].items():
        if answer is not None:
            assert sum(k['selected_masks'][kind]['joint_support'] for k in output['keys'])==answer['joint_support']
            assert sum(k['selected_masks'][kind]['separate_support'] for k in output['keys'])==answer['separate_support']
    all_sides=[s for pair in states for s in pair]
    output['features']=dict(
        keys=len(states),unknown=sum(s.unknown for s in all_sides),
        forced_discard=sum(s.forced_discard for s in all_sides),
        known_positives=sum(s.positive for s in all_sides),
        max_original_side=max(max(k['original_left_size'],k['original_right_size']) for k in output['keys']),
        max_unknown_side=max(s.unknown for s in all_sides),
        mean_effective_interval_width=statistics.mean(min(s.upper,s.positive+s.unknown)-max(s.lower,s.positive) for s in all_sides),
        total_effective_interval_width=sum(min(s.upper,s.positive+s.unknown)-max(s.lower,s.positive) for s in all_sides),
        max_finite_unknown_imbalance=max((k['unknown_imbalance'] for k in output['keys'] if k['unknown_imbalance'] is not None),default=None),
        one_side_exhausted_keys=sum((a.unknown==0) != (b.unknown==0) for a,b in states))
    return output


def summarize(rows):
    gaps=[r['same_evidence_discard_gap'] for r in rows if r['same_evidence_discard_gap'] is not None]
    joint_only=[r for r in rows if r['alternatives']['joint']['requires_coupling']]
    return dict(snapshots=len(rows),same_evidence_greedy_more=sum(g>0 for g in gaps),
        same_evidence_greedy_tie=sum(g==0 for g in gaps),
        same_evidence_greedy_fewer=sum(g<0 for g in gaps),
        min_discard_gap=min(gaps),max_discard_gap=max(gaps),
        separate_no_certified_superset=sum(r['alternatives']['separate'] is None for r in rows),
        joint_mask_requires_coupling=len(joint_only),
        fraction_requires_coupling=len(joint_only)/len(rows))


def compact(row):
    return {k:v for k,v in row.items() if k!='keys'}


def main(args):
    out=args.results.resolve()
    summary=json.loads((out/'summary.json').read_text())
    snapshots=[]
    rawbindings={}
    for entry in summary['rows']:
        for method,binding in entry['raw'].items():
            path=out/binding['path'];assert study.sha(path)==binding['sha256']
            record=study.read(path);rawbindings[binding['path']]=binding['sha256']
            snapshots.extend(one(s,record) for s in record['result']['sharp_snapshots'])
    actual_joint=[r for r in snapshots if r['trajectory']=='joint']
    actual_separate=[r for r in snapshots if r['trajectory']=='separate']
    initial=[r for r in actual_joint if r['index']==0]
    # Examples have a frozen rule in PROTOCOL.md and leave the full matrix intact.
    largest_support_gap=max(actual_joint,key=lambda r:
        r['alternatives']['joint']['separate_support']-r['alternatives']['joint']['joint_support'])
    smallest_greedy_gap=min((r for r in actual_joint
        if r['same_evidence_discard_gap'] is not None),key=lambda r:r['same_evidence_discard_gap'])
    largest_request_gain=max(summary['rows'],key=lambda r:r['request_gap'])
    statistics_output=dict(status='pass',source_id=summary['source_id'],raw_sha256=rawbindings,
        all=summarize(snapshots),joint_trajectory=summarize(actual_joint),
        separate_trajectory=summarize(actual_separate),initial=summarize(initial),
        joint_trajectory_groups={f'{key}_b{budget}':summarize([r for r in actual_joint
            if r['key']==key and r['budget']==budget]) for key in ('l1','l2') for budget in (125,500)},
        examples=dict(largest_same_mask_support_gap=largest_support_gap,
            smallest_same_evidence_greedy_gap=smallest_greedy_gap,
            largest_complete_request_gain=largest_request_gain),
        interpretation='Snapshots and budgets are repeated observations within 16 queries, '
            'not independent workloads. Same-evidence greedy results are distinct from '
            'certificate dominance and full-policy paid requests.')
    study.save(out/'snapshot_diagnostics.json.gz',dict(status='pass',rows=snapshots),True)
    study.save(out/'snapshot_analysis.json',statistics_output)
    study.save(out/'snapshot_overview.json',dict(status='pass',rows=[compact(r) for r in snapshots]))
    print(json.dumps({key:statistics_output[key] for key in ('all','joint_trajectory','separate_trajectory','initial')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=HERE/'results')
    main(p.parse_args())
