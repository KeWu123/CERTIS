"""Command-line entry point for the CERTIS research artifact."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'reproduction/study'
EXTERNAL = ROOT / 'reproduction/runtime/external_study'
GREEDY = ROOT / 'reproduction/runtime/greedy_count_profile'
sys.path[:0] = [str(STUDY), str(EXTERNAL), str(GREEDY), str(ROOT / 'code')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def check():
    from chain_certificate import check_frontier_vs_exact, check_exact_vs_bruteforce
    destination = ROOT / 'outputs/checks'
    destination.mkdir(parents=True, exist_ok=True)
    a = check_frontier_vs_exact()
    b = check_exact_vs_bruteforce()
    assert a['mismatch_count'] == 0 and b['bad_count'] == 0
    for path, extra in ((STUDY / 'check_marginal.py', []),
                        (GREEDY / 'checks_v3.py', [str(destination / 'execution.json')])):
        subprocess.run([sys.executable, str(path), *extra], cwd=ROOT, check=True)
    result = dict(status='pass', frontier=a, boolean_enumeration=b)
    (destination / 'core.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def verify():
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha(path) == expected['sha256'], relative
        assert path.stat().st_size == expected['bytes'], relative
    print(json.dumps(dict(status='pass', verified_files=len(manifest['files']))))


def normalize(value):
    from run_study import convert
    return json.loads(json.dumps(value, default=convert))


def smoke(output=None):
    import run_study as study
    from cached_greedy import execute as cached
    destination = (output or ROOT / 'outputs/smoke').resolve()
    cases = study.cases_for('constructed')[:1] + study.cases_for('native')[:2]
    cells = [dict(case_id=c.case_id, seed=101, budget=500) for c in cases]
    source_id = study.freeze(destination, cases, cells)
    checked = []
    for case in cases:
        reference = study.load_pinned_reference(case)
        methods = ['adaptive_continuous', 'fixed_continuous', 'joint_greedy'] if case.dataset!='goemotions_native' else ['adaptive_continuous','joint_greedy','batched_right']
        for method in methods:
            if method == 'joint_greedy':
                result = study.greedy(case,101,500)
            elif method == 'batched_right':
                result = study.batched_exact(case,method)
            else:
                result = study.marginal(case,101,500,method)
            seed,budget = (0,0) if method.startswith('batched') else (101,500)
            key = f'{case.case_id}__{method}__s{seed}__b{budget}'
            record = normalize(dict(key=key,case_id=case.case_id,method=method,seed=seed,
                budget=budget,source_id=source_id,status='pass',result=result,
                target_version=case.target_version,oracle_version=case.oracle_version,metadata=case.metadata))
            path=destination/'raw'/(key+'.json.gz')
            canonical='right_first' if method=='batched_right' else 'count_allocation_feedback_audit_degree'
            record['validation']=study.strict_check_record(record,case,record_path=path,expected_key=key,
                expected_method=method,expected_seed=seed,expected_budget=budget,
                expected_source_id=source_id,reference=reference,canonical_execution_method=canonical)
            cohort='native_v1' if case.dataset=='goemotions_native' else 'fixed_continuous_v2' if method=='fixed_continuous' else 'constructed_v1'
            if case.dataset!='goemotions_native' and method=='joint_greedy':
                historical=ROOT/'reproduction/runtime/greedy_count_profile/full/raw'/f'{case.case_id}__greedy_count_profile_feedback_audit_degree__s101__b500.json.gz'
            else:
                historical=STUDY/cohort/'raw'/(key+'.json.gz')
            expected=read(historical)['result']
            assert record['result']['oracle_ledger']==expected['oracle_ledger'], key
            for field in ('facts','discarded','regions'):
                assert record['result'][field]==expected[field], (key,field)
            assert record['result']['output']['full_count']==expected['output']['full_count']
            assert record['result']['output']['returned_count']==expected['output']['returned_count']
            if case.dataset=='goemotions_native' and method=='joint_greedy':
                fast=normalize(cached(case,101,500))
                for field in ('oracle_ledger','facts','discarded','events','regions','batches'):
                    assert fast[field]==record['result'][field],field
            study.save(path,record,True)
            row=dict(case_id=case.case_id,method=method,requests=len(record['result']['oracle_ledger']),strict_validation=True,ordered_ledger_matches_archive=True)
            checked.append(row)
            print(json.dumps(row),flush=True)
        case.close_sql()
    (destination/'summary.json').write_text(json.dumps(dict(status='pass',checks=checked),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','verify','smoke','analyze','plot','run','run-fixed','run-cached','run-greedy125'])
    parser.add_argument('--kind',choices=['constructed','native'],default='constructed')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.command=='check': check()
    elif args.command=='verify': verify()
    elif args.command=='smoke': smoke(args.output)
    elif args.command=='analyze': runpy.run_path(str(STUDY/'analyze.py'),run_name='__main__')
    elif args.command=='plot': runpy.run_path(str(ROOT/'code/plot_results.py'),run_name='__main__')
    elif args.command=='run':
        import run_study
        destination=(args.output or ROOT/'outputs'/args.kind).resolve()
        run_study.run(argparse.Namespace(kind=args.kind,output=str(destination),pilot=args.pilot))
    else:
        name={'run-fixed':'run_fixed_split_v2.py','run-cached':'run_cached.py','run-greedy125':'run_greedy125.py'}[args.command]
        destination=(args.output or ROOT/'outputs'/args.command).resolve()
        os.environ['CERTIS_OUTPUT']=str(destination)
        runpy.run_path(str(STUDY/name),run_name='__main__')


if __name__=='__main__':
    main()
