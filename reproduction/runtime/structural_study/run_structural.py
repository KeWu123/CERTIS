"""Complete-cost exploratory study. Freeze before checks or policy execution."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from structural_cases import HERE, EXTERNAL, SEEDS, build, digest, full_grid, geometry_grid, specification
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(0, str(HERE.parents[2] / 'code'))
from common import Oracle, evaluate, potential_degrees
from exact_baselines import run_exact
from extra_exact_controls import run_extra_exact
from direct_arms import make_regions, run_direct, Envelope
from cut_arms import run_cut
from count_allocation_arms import eligible, run_count_allocation, CountEnvelope
from bilinear_certificate import RegionSide, key_support

ROOT = HERE.parents[2]
METHODS = ('right', 'cover', 'dynamic_max_degree', 'Ddegree', 'C', 'Eall_mask')


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)): return sorted(jsonable(v) for v in x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Fraction): return str(x)
    return x


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def freeze():
    if (HERE / 'FREEZE.json').exists():
        verify_freeze(); return
    inputs = full_grid()
    geometry = geometry_grid()
    write_json(HERE / 'inputs.json', inputs)
    write_json(HERE / 'geometry_inputs.json', geometry)
    sources = [HERE / 'PROTOCOL.md', HERE / 'structural_cases.py', Path(__file__),
               HERE / 'inputs.json', HERE / 'geometry_inputs.json',
               *(EXTERNAL / f for f in ('common.py', 'exact_baselines.py', 'extra_exact_controls.py',
                  'direct_arms.py', 'cut_arms.py', 'count_allocation_arms.py')),
               *(ROOT / 'code' / f for f in ('bilinear_certificate.py', 'count_mask_planner.py',
                  'exact_hypergeom.py', 'relational_cut.py'))]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    for p in sources:
        dst = HERE / 'frozen_sources' / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
    write_json(HERE / 'FREEZE.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        classification='dated exploratory protocol; new evidence after existing studies, not original preregistration',
        source_sha256=hashes, sources_digest=digest(hashes), targets=len(inputs),
        geometry_targets=len(geometry), expected_policy_cells=594,
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         packages={k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scipy', 'duckdb')})))


def verify_freeze():
    frozen = json.loads((HERE / 'FREEZE.json').read_text(encoding='utf-8'))
    changed = [rel for rel, sha in frozen['source_sha256'].items()
               if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != sha]
    if changed: raise RuntimeError(f'Frozen source changed: {changed}')
    return frozen


def explicit_bag(spec, facts):
    return Counter((k, i, j) for k, (left, right) in enumerate(zip(spec['key_left'], spec['key_right']))
                   for i in left for j in right if facts.get(i) == facts.get(j) == 1)


def factorized_count(spec, facts):
    return sum(sum(facts.get(j) == 1 for j in left) * sum(facts.get(j) == 1 for j in right)
               for left, right in zip(spec['key_left'], spec['key_right']))


def run_one(case, method, seed, budget):
    if method in ('right', 'cover'): return run_exact(case, method)
    if method == 'dynamic_max_degree': return run_extra_exact(case, method)
    if method == 'Ddegree': return run_direct(case, seed, budget, audit=True, feedback=True, helper=False)
    if method == 'C': return run_cut(case, seed, budget, audit=True, feedback=True)
    if method == 'Eall_mask': return run_count_allocation(case, seed, budget, audit=True, feedback=True)
    raise ValueError(method)


def validate(spec, case, result, method, *, explicit=False):
    summary = result.get('summary', result)
    ledger = result.get('oracle_ledger', result.get('ledger'))
    helper = result['helper_ledger']; facts = result['facts']
    assert len(ledger) == len(facts) == summary['oracle_requests']
    assert len({r['id'] for r in ledger}) == len(ledger)
    assert {r['id']: r['label'] for r in ledger} == facts
    assert all(r['label'] == spec['truth'][r['id']] for r in ledger)
    for phase in ('audit', 'cut', 'execution'):
        entries = [r for r in ledger if r['phase'] == phase]
        assert len(entries) == summary['phase_counts'][phase]
    for prefix, entries in (('oracle', ledger), ('helper', helper)):
        assert summary[prefix + '_requests'] == len(entries)
        for direction in ('in', 'out'):
            assert summary[prefix + '_tokens_' + direction] == sum(r['tokens_' + direction] for r in entries)
            assert all(r['tokens_' + direction] == 1 for r in entries)
    assert len(helper) == (case.n if method == 'C' else 0)
    full = factorized_count(spec, dict(enumerate(spec['truth'])))
    returned = factorized_count(spec, facts)
    out = result['output']
    assert out['full_count'] == full and out['returned_count'] == returned
    assert out['lost_count'] == full - returned and out['bag_subset']
    if explicit:
        actual = Counter(case._sql['connection'].execute('SELECT key_id,left_id,right_id FROM returned_derivations').fetchall())
        assert actual == explicit_bag(spec, facts)
    coverage = []
    for record in result.get('regions', []):
        true_count = sum(spec['truth'][j] for j in record['ids'])
        coverage.append(dict(ids=record['ids'], actual_total=true_count,
                             lower=record['lower'], upper=record['upper'],
                             covered=record['lower'] <= true_count <= record['upper']))
    covered = all(r['covered'] for r in coverage)
    loss_violation = 10 * out['lost_count'] > full
    # Coverage failures and violations remain observations, never filtered.
    return dict(independent_factorized_full=full, independent_factorized_returned=returned,
                explicit_bag_checked=explicit, coverage=coverage,
                simultaneous_coverage=covered, coverage_failures=sum(not r['covered'] for r in coverage),
                loss_above_allowance=loss_violation,
                covered_but_loss_above_allowance=covered and loss_violation,
                ledger_checked=True, helper_charge_checked=True)


def tiny_checks():
    verify_freeze()
    checks = dict(generation_targets=0, complete_sql_policy_cells=0, audit_design_checks=0,
                  independent_explicit_bags=0, eligibility_rejections=0, helper_cost_checks=0,
                  independent_hypergeom_coverage_checks=0)
    for spec in full_grid() + geometry_grid():
        case = build(spec)
        assert case.n == spec['n']
        for side in (0, 1):
            ids = [j for j in range(case.n) if spec['side'][j] == side]
            assert sum(spec['secondary'][j] >= 0 for j in ids) == int(len(ids) * spec['reuse'])
        assert all(b < 0 or a != b for a, b in zip(spec['primary'], spec['secondary']))
        counts = [sum(j in ids for ids in (*case.key_left, *case.key_right)) for j in range(case.n)]
        assert all(c in (1, 2) for c in counts)
        assert eligible(case) == (spec['reuse'] == 0)
        twin = specification(spec['n'], spec['k'], spec['law'], 0., spec['seed'], geometry=spec['n'] <= 12)
        assert twin['primary'] == spec['primary'] and twin['truth'] == spec['truth']
        checks['generation_targets'] += 1
        case.close_sql()
    for reuse in (0., .5):
        base = specification(8, 2, 'uniform', reuse, 101, geometry=True)
        for truth in ([0] * 8, [1] * 8, [0, 1, 0, 1, 1, 0, 1, 0]):
            spec = {**base, 'truth': truth}
            case = build(spec)
            for method in METHODS:
                if method == 'Eall_mask' and reuse: continue
                result = run_one(case, method, 101, 1)
                checked = validate(spec, case, result, method, explicit=True)
                assert not checked['covered_but_loss_above_allowance']
                checks['complete_sql_policy_cells'] += 1
                checks['independent_explicit_bags'] += 1
                if method == 'C': checks['helper_cost_checks'] += 1
                if method == 'Ddegree' or method == 'Eall_mask':
                    rng = np.random.default_rng(101)
                    expected = []
                    for side in (0, 1):
                        expected += list(map(int, rng.choice(np.flatnonzero(case.side == side), size=1, replace=False)))
                    assert [r['id'] for r in result['oracle_ledger'] if r['phase'] == 'audit'] == expected
                    alpha_sum = sum((Fraction(r['alpha_executed']) for r in result['regions']), Fraction(0))
                    assert alpha_sum <= Fraction(1, 20)
                    checks['audit_design_checks'] += 1
                if method == 'C':
                    confidence = result['confidence']
                    cut_ids = set(result['cut_facts'])
                    for draw in confidence['side_draws']:
                        assert len(draw['sample_ids']) == min(1, draw['population'])
                        assert not cut_ids.intersection(draw['sample_ids'])
                    assert Fraction(confidence['delta_executed']) <= Fraction(1, 20)
                    checks['audit_design_checks'] += 1
            if reuse:
                try: CountEnvelope(case, {})
                except ValueError: checks['eligibility_rejections'] += 1
                else: raise AssertionError('Repeated identity planner was not rejected')
            # Oracle cache and score cache have separate one-time charges.
            oracle = Oracle(case)
            oracle.acquire((0, 0, 1), 'audit'); oracle.acquire((0, 1), 'execution')
            oracle.use_scores((0, 0, 1)); oracle.use_scores((1, 0))
            costs = oracle.summary()
            assert costs['oracle_requests'] == costs['helper_requests'] == 2
            assert costs['oracle_tokens_in'] == costs['helper_tokens_out'] == 2
            checks['helper_cost_checks'] += 1
            case.close_sql()
    # Independent exact finite-population probability, all N<=8 populations.
    from math import comb
    from exact_hypergeom import count_interval_exact
    for n in range(1, 9):
        for sample in range(n + 1):
            for total in range(n + 1):
                missed = Fraction(0)
                for positive in range(max(0, sample - n + total), min(sample, total) + 1):
                    lo, hi = count_interval_exact(n, sample, positive, .05)
                    if not lo <= total <= hi:
                        missed += Fraction(comb(total, positive) * comb(n - total, sample - positive), comb(n, sample))
                assert missed <= Fraction.from_float(.05)
                checks['independent_hypergeom_coverage_checks'] += 1
    write_json(HERE / 'checks.json', dict(status='pass', finished_utc=datetime.now(timezone.utc).isoformat(), **checks))
    print(json.dumps(checks), flush=True)


def run_grid():
    frozen = verify_freeze()
    assert json.loads((HERE / 'checks.json').read_text())['status'] == 'pass'
    specs = json.loads((HERE / 'inputs.json').read_text())
    rows = []
    raw = HERE / 'raw'; raw.mkdir(exist_ok=True)
    started = time.time()
    for target_no, spec in enumerate(specs, 1):
        case = build(spec)
        methods = METHODS if spec['reuse'] == 0 else METHODS[:-1]
        for method in methods:
            path = raw / f"{spec['target']}__{method}.json.gz"
            # Completed files can be resumed without rerunning or replacing cells.
            if path.exists():
                with gzip.open(path, 'rt', encoding='utf-8') as f: binding = json.load(f)
                rows.append(binding['row']); continue
            cpu0 = time.process_time(); wall0 = time.perf_counter()
            try:
                # Every cell pays independent SQL setup/reference/materialization CPU.
                case.close_sql()
                result = run_one(case, method, spec['seed'], max(1, spec['n'] // 16))
                checked = validate(spec, case, result, method)
                wall = time.perf_counter() - wall0; cpu = time.process_time() - cpu0
                summary = result.get('summary', result); out = result['output']
                row = {key: spec[key] for key in ('target', 'n', 'k', 'law', 'reuse', 'seed')}
                row.update(method=method, status='ok', oracle_requests=summary['oracle_requests'],
                           helper_requests=summary['helper_requests'],
                           complete_requests=summary['oracle_requests'] + summary['helper_requests'],
                           oracle_tokens=summary['oracle_tokens_in'] + summary['oracle_tokens_out'],
                           helper_tokens=summary['helper_tokens_in'] + summary['helper_tokens_out'],
                           complete_tokens=sum(summary[p + '_tokens_' + d] for p in ('oracle', 'helper') for d in ('in', 'out')),
                           audit_requests=summary['phase_counts']['audit'], cut_requests=summary['phase_counts']['cut'],
                           execution_requests=summary['phase_counts']['execution'],
                           full_count=out['full_count'], returned_count=out['returned_count'],
                           lost_count=out['lost_count'], actual_loss=out['loss'],
                           coverage_failures=checked['coverage_failures'],
                           simultaneous_coverage=checked['simultaneous_coverage'],
                           loss_above_allowance=checked['loss_above_allowance'],
                           covered_but_loss_above_allowance=checked['covered_but_loss_above_allowance'],
                           conflict_freeze=bool(result.get('conflict_freeze', False)),
                           cpu_seconds=cpu, wall_seconds=wall,
                           algorithm_seconds=result.get('timings', {}).get('algorithm_seconds', ''),
                           mode=result.get('mode', 'exact'), error='')
                binding = dict(source_digest=frozen['sources_digest'], input_sha256=spec['input_sha256'],
                               row=row, independent_validation=checked, result=result)
            except Exception as exc:
                row = {key: spec[key] for key in ('target', 'n', 'k', 'law', 'reuse', 'seed')}
                row.update(method=method, status='failure', error=repr(exc), cpu_seconds=time.process_time()-cpu0,
                           wall_seconds=time.perf_counter()-wall0)
                binding = dict(source_digest=frozen['sources_digest'], input_sha256=spec['input_sha256'],
                               row=row, exception_traceback=traceback.format_exc())
            with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump(jsonable(binding), f, ensure_ascii=False)
            rows.append(row)
            print(f"{len(rows):03d}/594 {target_no:03d}/108 {spec['target']} {method} {row['status']} "
                  f"req={row.get('complete_requests', '?')} cpu={row['cpu_seconds']:.3f} elapsed={time.time()-started:.1f}", flush=True)
        case.close_sql()
        # Normalize fields if failures occurred; no cell disappears.
        fields = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(HERE / 'cells.csv', [{k: row.get(k, '') for k in fields} for row in rows])
        write_json(HERE / 'progress.json', dict(completed_cells=len(rows), expected_cells=594,
                   targets_completed=target_no, elapsed_seconds=time.time()-started))
    assert len(rows) == 594
    summarize(rows)


def geometry():
    verify_freeze()
    records = []
    for spec in json.loads((HERE / 'geometry_inputs.json').read_text()):
        n = spec['n']; truth = np.asarray(spec['truth'], dtype=np.int64)
        worlds = np.asarray(list(product((0, 1), repeat=n)), dtype=np.int64)
        order = sorted(range(n), key=lambda j: (sum(len(right) if j in left else len(left) if j in right else 0
                       for left, right in zip(spec['key_left'], spec['key_right'])), j))
        masks = [('empty', set()), ('full', set(range(n))), ('left', set(range(n//2))), ('right', set(range(n//2, n)))]
        masks += [(f'degree_prefix_{q}', set(order[:n*q//4])) for q in (1, 2, 3)]
        rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], n, spec['k'], 71]))
        masks += [(f'random_{q}', set(map(int, rng.permutation(n)[:n//2]))) for q in range(4)]
        cache_order = rng.permutation(n)
        for cache_fraction in (0., .25, .5):
            cache = set(map(int, cache_order[:int(n * cache_fraction)]))
            facts = {j: int(truth[j]) for j in cache}
            for slack in (0, 1):
                intervals = {}
                feasible = np.ones(len(worlds), dtype=bool)
                for j, y in facts.items(): feasible &= worlds[:, j] == y
                for ids in (*spec['key_left'], *spec['key_right']):
                    ids = tuple(ids); total = int(truth[list(ids)].sum())
                    lo, hi = max(0, total-slack), min(len(ids), total+slack)
                    intervals[ids] = (lo, hi)
                    counts = worlds[:, list(ids)].sum(axis=1)
                    feasible &= (lo <= counts) & (counts <= hi)
                possible = worlds[feasible]
                assert len(possible) and any(np.array_equal(z, truth) for z in possible)
                total_counts = np.zeros(len(possible), dtype=np.int64)
                for left, right in zip(spec['key_left'], spec['key_right']):
                    total_counts += possible[:, left].sum(axis=1) * possible[:, right].sum(axis=1)
                for name, base_mask in masks:
                    mask = base_mask - cache
                    returned_counts = np.zeros(len(possible), dtype=np.int64)
                    copied = 0
                    for left, right in zip(spec['key_left'], spec['key_right']):
                        retained_left = [j for j in left if j not in mask]
                        retained_right = [j for j in right if j not in mask]
                        returned_counts += possible[:, retained_left].sum(axis=1) * possible[:, retained_right].sum(axis=1)
                        sides = []
                        for ids in (left, right):
                            unknown = set(ids) - cache; lost = len(unknown & mask)
                            h = sum(facts.get(j) == 1 for j in ids)
                            lo, hi = intervals[tuple(ids)]
                            sides.append(RegionSide(lo, hi, h, lost, len(unknown)-lost))
                        copied += key_support(*sides, Fraction(1, 10))
                    boolean = int(np.max(9 * total_counts - 10 * returned_counts))
                    assert copied >= boolean
                    if not spec['reuse']: assert copied == boolean
                    records.append(dict(target=spec['target'], n=n, k=spec['k'], reuse=spec['reuse'], seed=spec['seed'],
                        cache_fraction=cache_fraction, interval_slack=slack, mask=name,
                        cached_ids=sorted(cache), effective_discarded=sorted(mask), feasible_worlds=len(possible),
                        boolean_scaled_support=boolean, copied_scaled_support=copied, gap=copied-boolean,
                        boolean_certified=boolean <= 0, copied_certified=copied <= 0))
    write_json(HERE / 'geometry_records.json', records)
    axis = []
    for reuse in (0., .25, .5, 1.):
        subset = [r for r in records if r['reuse'] == reuse]
        axis.append(dict(reuse=reuse, cases=len(subset), strict_gap_cases=sum(r['gap'] > 0 for r in subset),
                         mean_gap=float(np.mean([r['gap'] for r in subset])), max_gap=max(r['gap'] for r in subset),
                         missed_certifications=sum(r['boolean_certified'] and not r['copied_certified'] for r in subset)))
    write_csv(HERE / 'geometry_summary.csv', axis)
    write_json(HERE / 'geometry_validation.json', dict(status='pass', target_count=48, mask_evidence_cases=len(records),
        statistical_audit=False, classification='explicitly covered constructed intervals; geometry only', summary=axis))
    print(json.dumps(axis), flush=True)


def summarize(rows=None):
    if rows is None:
        rows = []
        for path in sorted((HERE / 'raw').glob('*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf-8') as f: rows.append(json.load(f)['row'])
    successes = [r for r in rows if r['status'] == 'ok']
    aggregations = []
    for axis in ('overall', 'n', 'k', 'law', 'reuse'):
        for value in (['all'] if axis == 'overall' else sorted(set(r[axis] for r in rows))):
            for method in METHODS:
                subset = [r for r in successes if r['method'] == method and (axis == 'overall' or r[axis] == value)]
                attempted = [r for r in rows if r['method'] == method and (axis == 'overall' or r[axis] == value)]
                if not attempted: continue
                row = dict(axis=axis, value=value, method=method, attempted=len(attempted), success=len(subset),
                           failures=len(attempted)-len(subset))
                for key in ('oracle_requests', 'helper_requests', 'complete_requests', 'oracle_tokens', 'helper_tokens',
                            'complete_tokens', 'cpu_seconds', 'wall_seconds', 'actual_loss'):
                    vals = [r[key] for r in subset]
                    row['mean_' + key] = float(np.mean(vals)) if vals else ''
                    row['max_' + key] = max(vals) if vals else ''
                for key in ('coverage_failures', 'loss_above_allowance', 'covered_but_loss_above_allowance', 'conflict_freeze'):
                    row[key] = sum(r[key] for r in subset)
                aggregations.append(row)
    write_csv(HERE / 'axis_summary.csv', aggregations)
    paired = []
    by_target = defaultdict(dict)
    for row in successes: by_target[row['target']][row['method']] = row
    for target, methods in by_target.items():
        for method in ('Ddegree', 'C', 'Eall_mask'):
            if method not in methods: continue
            exact = [methods[m] for m in ('right', 'cover', 'dynamic_max_degree')]
            reference = min(exact, key=lambda r: (r['complete_requests'], r['method']))
            row = methods[method]
            paired.append(dict(target=target, method=method, n=row['n'], k=row['k'], law=row['law'], reuse=row['reuse'], seed=row['seed'],
                comparator='posthoc minimum of three actually executed exact controls (descriptive only)', best_exact_method=reference['method'],
                best_exact_requests=reference['complete_requests'], complete_requests=row['complete_requests'],
                oracle_requests=row['oracle_requests'], helper_requests=row['helper_requests'],
                saving=reference['complete_requests']-row['complete_requests'], actual_loss=row['actual_loss']))
    write_csv(HERE / 'paired_comparisons.csv', paired)
    # Independent post-run raw inspection: source/input binding, ledger arithmetic,
    # SQL statement evidence, exact cell roster, and no hidden target replacement.
    specs = {s['target']: s for s in json.loads((HERE / 'inputs.json').read_text())}
    frozen = verify_freeze(); seen = set(); raw_inspection_failures = []
    for path in (HERE / 'raw').glob('*.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f: binding = json.load(f)
        row = binding['row']; spec = specs[row['target']]
        try:
            assert binding['source_digest'] == frozen['sources_digest']
            assert binding['input_sha256'] == spec['input_sha256']
            assert (row['target'], row['method']) not in seen
            seen.add((row['target'], row['method']))
            if row['status'] != 'ok': continue
            result = binding['result']; summary = result.get('summary', result)
            ledger = result.get('oracle_ledger', result.get('ledger'))
            assert row['oracle_requests'] == len(ledger) == summary['oracle_requests']
            assert all(spec['truth'][r['id']] == r['label'] for r in ledger)
            assert row['complete_tokens'] == 2 * row['complete_requests']
            assert 'CREATE OR REPLACE TEMP TABLE returned_derivations' in result['output']['returned_sql']
            facts = {int(j): y for j, y in result['facts'].items()}
            assert factorized_count(spec, facts) == row['returned_count']
        except Exception as exc: raw_inspection_failures.append(dict(path=str(path), error=repr(exc)))
    expected = {(target, method) for target, spec in specs.items()
                for method in (METHODS if spec['reuse'] == 0 else METHODS[:-1])}
    write_json(HERE / 'raw_inspection.json', dict(status='pass' if seen == expected and not raw_inspection_failures else 'failure',
        inspected=len(seen), expected=len(expected), missing=sorted(expected-seen), unexpected=sorted(seen-expected),
        inspection_failures=raw_inspection_failures))
    overall = [r for r in aggregations if r['axis'] == 'overall']
    lines = ['# Structural study: complete-cost exploratory results', '',
        'Frozen on 2026-10-16 before policy execution. Fixed synthetic labels and unit request prices; no live model inference or monetary cost claim.', '',
        f"Executed {len(rows)}/594 cells across {len(specs)}/108 targets; {len(successes)} actual DuckDB output materializations. Failures: {len(rows)-len(successes)}.", '',
        'Every oracle and helper request has one synthetic input token and one synthetic output token. C purchases all constant .5 helper scores; the signal is artificial and uninformative. CPU includes setup, reference SQL, actual returned bag materialization, validation, and policy execution. It excludes serialization and input generation. Each cell gets a fresh SQL connection.', '',
        '| Method | Cells | Oracle requests (mean) | Helper requests (mean) | Complete requests (mean) | CPU s (mean) | Mean loss | Max loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in overall:
        lines.append(f"| {r['method']} | {r['success']} | {r['mean_oracle_requests']:.2f} | {r['mean_helper_requests']:.2f} | {r['mean_complete_requests']:.2f} | {r['mean_cpu_seconds']:.4f} | {r['mean_actual_loss']:.5f} | {r['max_actual_loss']:.5f} |")
    lines += ['', 'Eall_mask covers only the 54 key-disjoint targets. Its overall row has a different target population. Use paired_comparisons.csv and axis_summary.csv for like-for-like comparisons. The posthoc minimum exact comparator is descriptive, not an executable oracle policy.', '',
              f"Observed region coverage failures: {sum(r['coverage_failures'] for r in successes)}; cells with loss above 0.1: {sum(r['loss_above_allowance'] for r in successes)}; covered cells above 0.1: {sum(r['covered_but_loss_above_allowance'] for r in successes)}. Confidence failures are retained in raw bindings and cell tables.", '']
    for method in ('Ddegree', 'C', 'Eall_mask'):
        subset = [r for r in paired if r['method'] == method]
        lines.append(f"{method}: complete requests beat/tie/lose to the descriptive best executed exact control in {sum(r['saving']>0 for r in subset)}/{sum(r['saving']==0 for r in subset)}/{sum(r['saving']<0 for r in subset)} of {len(subset)} paired targets; mean saving {np.mean([r['saving'] for r in subset]):.2f} requests.")
    lines += ['', 'The design isolates total identity count N, nominal key count K, deterministic key-size law, and secondary membership while pairing primary assignments and labels across reuse. Effective nonempty keys and full answer cardinality may change with these axes. Constant request prices isolate call savings; token totals are synthetic accounting, not estimated inference costs.', '',
              'Algorithm and cost provenance: frozen_sources/ and FREEZE.json pin common.Oracle, actual DuckDB evaluator, exact controls, Ddegree, C, Eall_mask, hypergeometric inversion, integer support, exact count DP, compressed cut implementation, generator, inputs, and this driver. Runtime versions are in FREEZE.json.', '',
              'Geometry results use covered, constructed exact-count or ±1 intervals, cache fractions 0/.25/.5, 11 predetermined masks, and all Boolean worlds. They are not statistical audits. geometry_records.json retains each mask and cache, copied support, global Boolean support, and sharpness gap.']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(cells=len(rows), successful=len(successes), materialized=len(successes),
                         coverage_failures=sum(r['coverage_failures'] for r in successes),
                         loss_violations=sum(r['loss_above_allowance'] for r in successes),
                         raw_inspection_failures=raw_inspection_failures)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'checks', 'grid', 'geometry', 'summarize'))
    action = parser.parse_args().action
    {'freeze': freeze, 'checks': tiny_checks, 'grid': run_grid, 'geometry': geometry, 'summarize': summarize}[action]()
