"""Post hoc initial-cache certification diagnostic for all eight live Qwen cells.

Run only after the independent native analyzer accepts all 32 production runs.
This script never invokes an oracle, reruns a policy, or modifies measured cost.
Heavy imports and exact profiles are delayed until the complete-analysis gate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RELEASE = WORKSPACE
SHARP = WORKSPACE/'reproduction/sharp_value'
METHODS = ('batched_right', 'adaptive_continuous', 'separate', 'joint')
SEEDS = (101, 202)


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def assert_close(left, right):
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7), (left, right)


def write_frozen_json(path, value):
    """Keep any previously written diagnostic immutable unless bytes match."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode('utf-8')
    if path.exists():
        assert path.read_bytes() == payload, ('Existing diagnostic differs', str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        assert not temporary.exists(), temporary
        temporary.write_bytes(payload)
        temporary.replace(path)


def complete_gate(source, analysis_path):
    """Check completeness and raw bindings before importing profile machinery."""
    analysis = read(analysis_path)
    assert analysis['status'] == 'pass'
    assert analysis['validated_runs'] == 32 and analysis['remaining_runs'] == 0
    assert analysis['execution_sequence_checked']
    assert analysis['all_paid_model_labels_rechecked']
    assert analysis['independent_factorized_bag_counts']
    assert analysis['returned_first_sql']
    manifest = read(source/'manifest.json')
    assert jsha({k:v for k,v in manifest.items() if k != 'binding'}) == manifest['binding']
    assert analysis['binding'] == manifest['binding']
    assert manifest['rows_per_side'] == 5000 and manifest['audit_budget'] == 500
    assert tuple(manifest['methods']) == METHODS and tuple(manifest['seeds']) == SEEDS
    assert manifest['recall'] == '9/10' and manifest['delta'] == '1/20'
    assert not manifest['original_70b_labels_used_for_policy']
    assert len(manifest['cases']) == len(set(manifest['cases'])) == 4
    expected_cases = {
        'dbpedia_native__agent__person__l1', 'dbpedia_native__agent__person__l2',
        'dbpedia_native__person__athlete__l1', 'dbpedia_native__person__athlete__l2'}
    assert set(manifest['cases']) == expected_cases
    summary = read(source/'summary.json')
    assert summary['status'] == 'pass' and summary['verified'] == summary['expected'] == 32
    assert summary['binding'] == manifest['binding']
    assert summary['live_outputs_used_for_policy'] and summary['fixed_reference_disagreements'] == 0
    assert not (source/'oracle_disagreement.json').exists()
    expected = {(case, method, seed) for case in manifest['cases']
        for method in METHODS for seed in SEEDS}
    assert len(analysis['rows']) == 32 and len(analysis['raw_sha256']) == 32
    lookup = {(r['case_id'], r['method'], r['seed']):r for r in analysis['rows']}
    assert set(lookup) == expected
    assert len(summary['rows']) == 32
    summary_lookup = {(r['case_id'], r['method'], r['seed']):r for r in summary['rows']}
    assert set(summary_lookup) == expected
    raw_paths = sorted((source/'raw').glob('*.json.gz'))
    assert len(raw_paths) == 32
    records = {}
    for path in raw_paths:
        assert sha(path) == analysis['raw_sha256'][path.name], path.name
        record = read(path)
        row = record['row']
        key = (row['case_id'], row['method'], row['seed'])
        assert key in expected and key not in records
        assert record['status'] == 'pass' and record['binding'] == manifest['binding']
        assert row == summary_lookup[key]
        for field in ('requests', 'complete_seconds', 'inference_seconds', 'local_seconds',
                'returned_sql_seconds', 'audit_budget', 'coverage', 'conflict'):
            assert row[field] == lookup[key][field], (key, field)
        records[key] = (record, path)
    assert set(records) == expected
    for relative, expected_sha in manifest['sources'].items():
        path = WORKSPACE/relative.replace('\\', '/')
        frozen = source/'source_versions'/(expected_sha+Path(relative).suffix)
        if path.is_file():
            assert sha(path) == expected_sha, relative
        assert frozen.is_file() and sha(frozen) == expected_sha, relative
    return manifest, analysis, lookup, records


def initial_state(result, groups, SideCounts):
    """Use audit labels only; never read unqueried or later execution labels."""
    ledger = result['oracle_ledger']
    audit = [row for row in ledger if row['phase'] == 'audit']
    assert len(audit) == 1000 and ledger[:1000] == audit
    audit_facts = {int(row['id']):int(row['label']) for row in audit}
    assert len(audit_facts) == 1000 and set(audit_facts.values()) <= {0, 1}
    assert all(0 <= identity < 10000 for identity in audit_facts)
    assert sum(identity < 5000 for identity in audit_facts) == 500
    all_unknown = set(range(10000))-audit_facts.keys()
    retained = set(map(int, result['initial_retained_unknown']))
    assert retained <= all_unknown
    mask = all_unknown-retained
    assert mask == set(map(int, result['initial_discarded']))
    assert len(mask) == result['initial_certified_discard_count']
    regions = {tuple(map(int, r['ids'])):r for r in result['regions']}
    assert len(regions) == len(result['regions']) == 2*len(groups)
    states, counts, key_rows = [], [], []
    incident = set()
    expected_domains = set()
    for name, members in groups.items():
        pair = (tuple(map(int, members)), tuple(int(j)+5000 for j in members))
        side_states, side_counts, sides = [], [], []
        for ids in pair:
            incident.update(ids)
            expected_domains.add(ids)
            region = regions[ids]
            unknown = tuple(j for j in ids if j not in audit_facts)
            h = sum(audit_facts.get(j) == 1 for j in ids)
            assert region['population'] == len(ids)
            assert region['sample'] == sum(j in audit_facts for j in ids)
            assert region['positive_sample'] == h
            side = SideCounts(int(region['lower']), int(region['upper']), h, len(unknown), 0)
            discarded = sum(j in mask for j in unknown)
            side_states.append(side)
            side_counts.append(discarded)
            sides.append(dict(lower=side.lower, upper=side.upper, positive=side.positive,
                unknown=side.unknown, forced_discard=0, actual_discarded_unknown=discarded,
                original_population=len(ids), sampled=region['sample']))
        states.append(tuple(side_states))
        counts.append(tuple(side_counts))
        key_rows.append(dict(key=str(name), left=sides[0], right=sides[1]))
    assert set(regions) == expected_domains
    orphan_unknown = all_unknown-incident
    # Full 5k native category memberships contain every identity exactly once.
    assert incident == set(range(10000)) and not orphan_unknown
    assert sum(sum(pair) for pair in counts) == len(mask)
    event = result['events'][0]
    assert event['paid'] == 1000 and event['discard_cardinality'] == len(mask)
    assert tuple(tuple(map(int, p)) for p in event['chosen_counts']) == tuple(counts)
    assert event['chosen_unknown_discard_count'] == len(mask)
    assert event['chosen_incident_unknown_discard_count'] == len(mask)
    return dict(keys=tuple(states), counts=tuple(counts), mask=mask,
        audit_facts=audit_facts, audit=audit, unknown=all_unknown,
        orphan_unknown=orphan_unknown, key_rows=key_rows, event=event)


def main(args):
    source, analysis_path = args.source.resolve(), args.analysis.resolve()
    output = args.output.resolve()
    provenance_output = args.provenance.resolve() if args.provenance else output.with_name(
        output.stem+'_provenance.json')
    manifest, analysis, checked_rows, records = complete_gate(source, analysis_path)

    # All imports below are deliberately after the complete checked 32-run gate.
    import pandas as pd
    sys.path[:0] = [str(SHARP), str(RELEASE/'code')]
    from sharp_wrapper import build_profile, mask_support, exact
    from fractional_bound import fractional_upper_bound
    from bilinear_certificate import RegionSide, key_support

    # Bind the verified analysis algorithm independently of production sources.
    fractional_provenance = read(SHARP/'results/fractional_provenance.json')
    fractional_source_sha = fractional_provenance['sources']['fractional_bound.py']
    historical_fractional = SHARP/'source_versions'/(fractional_source_sha+'.py')
    assert historical_fractional.is_file() and sha(historical_fractional) == fractional_source_sha
    fractional_mapping_path = SHARP/'portable_sources.json'
    fractional_portable_mapping = next(row for row in read(fractional_mapping_path)
        if row['path'] == 'reproduction/sharp_value/fractional_bound.py')
    assert fractional_portable_mapping['original_sha256'] == fractional_source_sha
    assert fractional_portable_mapping['portable_sha256'] == sha(SHARP/'fractional_bound.py')
    assert (WORKSPACE/fractional_portable_mapping['frozen_path']).resolve() == historical_fractional.resolve()
    assert sha(WORKSPACE/fractional_portable_mapping['frozen_path']) == fractional_source_sha
    dataset = RELEASE/'reproduction/runtime/candidate_systems/external/semviews/data/datasets/dbpedia.parquet'
    assert sha(dataset) == manifest['data_sha256']
    frame = pd.read_parquet(dataset).sort_values('row_id').reset_index(drop=True)
    roster = list(map(str, frame.row_id))
    assert roster == manifest['roster'] and len(roster) == len(set(roster)) == 5000
    allowance = Fraction(1, 10)
    rows, used_raw = [], {}
    for case_id in manifest['cases']:
        _, left_predicate, right_predicate, column = case_id.split('__')
        assert column in ('l1', 'l2') and frame[column].notna().all()
        groups = frame.groupby(column, sort=True).indices
        identities = ['dbpedia/'+p+'|'+rid for p in (left_predicate, right_predicate) for rid in roster]
        for seed in SEEDS:
            jrecord, jpath = records[(case_id, 'joint', seed)]
            srecord, spath = records[(case_id, 'separate', seed)]
            joint_result, separate_result = jrecord['result'], srecord['result']
            for result in (joint_result, separate_result):
                assert Fraction(result['allowance']) == allowance and Fraction(result['delta']) == Fraction(1, 20)
                assert result['audit_budget'] == 500 and not result['diagnostic_instrumentation']
                for paid in result['oracle_ledger'][:1000]:
                    assert paid['identity'] == identities[int(paid['id'])]
            joint = initial_state(joint_result, groups, exact.SideCounts)
            separate = initial_state(separate_result, groups, exact.SideCounts)
            assert [(r['id'], r['label']) for r in joint['audit']] == [
                (r['id'], r['label']) for r in separate['audit']]
            assert joint_result['regions'] == separate_result['regions']
            assert joint['keys'] == separate['keys']
            keys = joint['keys']
            hj = mask_support(keys, joint['counts'], allowance)
            hs = mask_support(keys, joint['counts'], allowance, True)
            assert hj <= hs and hj <= 0
            assert hj == joint['event']['scaled_support']
            # Recheck using the original integral-polygon kernel as production did.
            direct = 0
            for (a,b),(dl,dr) in zip(keys,joint['counts']):
                direct += key_support(RegionSide(a.lower,a.upper,a.positive,dl,a.unknown-dl),
                    RegionSide(b.lower,b.upper,b.positive,dr,b.unknown-dr), allowance)
            assert direct == hj
            first_observation = joint_result['policy_support_observations'][0]
            assert first_observation['policy_support'] == first_observation['joint_support'] == hj
            assert first_observation['separate_support'] is None
            sep_hj = mask_support(keys, separate['counts'], allowance)
            sep_hs = mask_support(keys, separate['counts'], allowance, True)
            assert sep_hj <= sep_hs <= 0 and sep_hj == separate['event']['scaled_support']
            sep_observation = separate_result['policy_support_observations'][0]
            assert sep_observation['joint_support'] == sep_hj
            assert sep_observation['separate_support'] == sep_observation['policy_support'] == sep_hs
            profiles = tuple(build_profile(a,b,allowance,separate=True) for a,b in keys)
            assert all(profile.forced == 0 for profile in profiles)
            upper = fractional_upper_bound(profiles)
            assert isinstance(upper, Fraction) and upper >= 0
            incident_upper = upper.numerator//upper.denominator
            orphan_count = len(joint['orphan_unknown'])
            total_upper = incident_upper+orphan_count
            gj = sum(sum(pair) for pair in joint['counts'])
            gs = sum(sum(pair) for pair in separate['counts'])
            assert gs <= incident_upper
            jr, sr = checked_rows[(case_id,'joint',seed)], checked_rows[(case_id,'separate',seed)]
            for checked, record in ((jr,jrecord),(sr,srecord)):
                assert checked['requests'] == len(record['result']['oracle_ledger'])
                assert_close(checked['complete_seconds'],record['row']['complete_seconds'])
            rows.append(dict(case_id=case_id,key=column,seed=seed,audit_budget_per_side=500,
                oracle_scope='Fresh live Qwen2.5-7B-Q4_K_M labels only; no 70B labels used',
                initial_cache=dict(audit_paid_identities=1000,unknown_identities=len(joint['unknown']),
                    forced_incident_discards=0,orphan_unknown_count=orphan_count,
                    audit_facts={str(k):v for k,v in sorted(joint['audit_facts'].items())},
                    audit_facts_sha256=jsha(sorted(joint['audit_facts'].items()))),
                actual_joint_initial=dict(incident_discard_count=gj,total_unknown_discard_count=len(joint['mask']),
                    discarded_ids=sorted(joint['mask']),counts=joint['counts'],joint_support=hj,
                    separate_support=hs,only_joint_certifies=hj <= 0 < hs,
                    initial_event_support_checked=True,original_polygon_support_checked=True),
                actual_separate_initial=dict(incident_discard_count=gs,total_unknown_discard_count=len(separate['mask']),
                    discarded_ids=sorted(separate['mask']),counts=separate['counts'],joint_support=sep_hj,
                    separate_support=sep_hs),
                separate_fractional_relaxation=dict(incident_discard_upper_bound_rational=str(upper),
                    numerator=upper.numerator,denominator=upper.denominator,
                    incident_integer_discard_upper_bound=incident_upper,
                    total_integer_discard_upper_bound=total_upper,
                    joint_vs_integer_separate_proved_advantage_at_least=max(0,gj-incident_upper),
                    heap_independent_separation=gj > incident_upper,
                    interpretation='Upper bound on the separate certificate fixed-cache discard optimum; '
                        'fractional choices are not executable masks or adaptive request optima.'),
                actual_initial_greedy_discard_gap=gj-gs,
                complete_measured_costs=dict(
                    joint=dict(requests=jr['requests'],complete_seconds=jr['complete_seconds'],
                        inference_seconds=jr['inference_seconds'],local_seconds=jr['local_seconds']),
                    separate=dict(requests=sr['requests'],complete_seconds=sr['complete_seconds'],
                        inference_seconds=sr['inference_seconds'],local_seconds=sr['local_seconds']),
                    request_gap=sr['requests']-jr['requests'],
                    complete_seconds_gap=sr['complete_seconds']-jr['complete_seconds'],
                    request_saving_percent=100*(1-jr['requests']/sr['requests']),
                    complete_time_saving_percent=100*(1-jr['complete_seconds']/sr['complete_seconds']),
                    scope='The actual matched production rows; positive gaps favor joint. '
                        'This diagnostic adds no time to either production cost.'),
                actual_joint_coverage=jr['coverage'],actual_separate_coverage=sr['coverage'],
                key_count_states=joint['key_rows'],raw_sha256=dict(joint=sha(jpath),separate=sha(spath))))
            used_raw[jpath.name], used_raw[spath.name] = sha(jpath), sha(spath)
    assert len(rows) == 8 and len(used_raw) == 16
    assert {(r['case_id'],r['seed']) for r in rows} == {
        (case,seed) for case in manifest['cases'] for seed in SEEDS}
    diagnostic_sources = {str(path.relative_to(WORKSPACE)).replace('\\','/'):sha(path)
        for path in (Path(__file__).resolve(),SHARP/'sharp_wrapper.py',SHARP/'fractional_bound.py',
            SHARP/'results/fractional_provenance.json',HERE/'analyze_native_live.py',
            HERE/'LIVE_INITIAL_DIAGNOSTIC.md')}
    provenance = dict(production_binding=manifest['binding'],
        production_manifest_sha256=sha(source/'manifest.json'),
        production_summary_sha256=sha(source/'summary.json'),
        qwen_reference_binding_sha256=sha(source/'reference_binding.json'),
        checked_analysis_sha256=sha(analysis_path),checked_analysis_status='pass',
        checked_production_runs=32,dataset_sha256=sha(dataset),
        production_source_sha256=manifest['sources'],diagnostic_source_sha256=diagnostic_sources,
        checked_raw_sha256=analysis['raw_sha256'],used_joint_separate_raw_sha256=used_raw)
    provenance['fractional_function_portability'] = dict(
        verified_historical_source_sha256=fractional_source_sha,
        historical_source_path=str(historical_fractional.relative_to(WORKSPACE)),
        current_portable_source_sha256=sha(SHARP/'fractional_bound.py'),
        portability_manifest_sha256=sha(fractional_mapping_path),
        portability_changes=fractional_portable_mapping['portability_changes'],
        scope='Historical verification refers to the frozen original bytes; '
            'diagnostic_source_sha256 records current portable diagnostic dependencies.')
    result = dict(status='pass',secondary_post_hoc_diagnostic=True,
        scope='All eight actual joint cells: four related native queries by two audit seeds. '
            'Initial fixed-cache masks use only the 1000 paid audit facts; no final facts are reused. '
            'Count evidence is the declared uncoupled side/key intervals plus acquired audit facts. '
            'The same-Qwen cost linkage is descriptive and does not prove an adaptive request optimum.',
        reconstruction='allUnknownAfterAudit minus result.initial_retained_unknown; checked against '
            'initial_discarded, initial_certified_discard_count and first event counts/support',
        aggregation=dict(queries=4,audit_seeds=list(SEEDS),cells=8,
            only_joint_certifies_cells=sum(r['actual_joint_initial']['only_joint_certifies'] for r in rows),
            proved_cardinality_separation_cells=sum(r['separate_fractional_relaxation']['heap_independent_separation'] for r in rows),
            max_proved_cardinality_advantage=max(r['separate_fractional_relaxation'][
                'joint_vs_integer_separate_proved_advantage_at_least'] for r in rows)),
        provenance=provenance,rows=rows)
    write_frozen_json(output,result)
    write_frozen_json(provenance_output,dict(status='pass',secondary_post_hoc_diagnostic=True,
        diagnostic_output_sha256=sha(output),**provenance))
    print(json.dumps(dict(status='pass',cells=8,output=str(output),provenance=str(provenance_output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'results')
    parser.add_argument('--analysis',type=Path,default=HERE/'native_live_analysis.json')
    parser.add_argument('--output',type=Path,default=HERE.parents[1]/'outputs/live_initial_diagnostic.json')
    parser.add_argument('--provenance',type=Path)
    main(parser.parse_args())
