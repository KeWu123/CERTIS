"""Post-run descriptive analysis; does not execute or alter any policy."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent


def write_csv(name, rows):
    if not rows: return
    with (HERE / name).open('w', newline='', encoding='utf-8-sig') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)


def main():
    specs = {s['target']: s for s in json.loads((HERE / 'inputs.json').read_text(encoding='utf-8'))}
    descriptors, bindings, failures, axis_rows = [], [], [], []
    for s in specs.values():
        degrees = np.zeros(s['n'], dtype=np.int64)
        for left, right in zip(s['key_left'], s['key_right']):
            degrees[left] += len(right); degrees[right] += len(left)
        labels = np.asarray(s['truth'])
        row = {k: s[k] for k in ('target', 'n', 'k', 'law', 'reuse', 'seed')}
        row.update(primary_nonempty_keys=sum(x > 0 for x in s['primary_counts_per_side']),
                   productive_incidence_keys=sum(bool(left) and bool(right) for left, right in zip(s['key_left'], s['key_right'])),
                   total_incidences=sum(map(len, (*s['key_left'], *s['key_right']))),
                   reused_identities=sum(x >= 0 for x in s['secondary']),
                   structural_pair_derivations=int(degrees.sum() // 2),
                   mean_degree=float(np.mean(degrees)), max_degree=int(np.max(degrees)),
                   degree_cv=float(np.std(degrees) / np.mean(degrees)) if degrees.sum() else 0.,
                   positive_rate=float(labels.mean()), positive_left=float(labels[:s['n']//2].mean()),
                   positive_right=float(labels[s['n']//2:].mean()))
        descriptors.append(row)
    write_csv('target_descriptors.csv', descriptors)
    for path in sorted((HERE / 'raw').glob('*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as f: b = json.load(f)
        bindings.append(b)
        for index, r in enumerate(b.get('independent_validation', {}).get('coverage', [])):
            if not r['covered']:
                row = {k: b['row'][k] for k in ('target', 'n', 'k', 'law', 'reuse', 'seed', 'method')}
                row.update(region_index=index, region_population=len(r['ids']), actual_total=r['actual_total'],
                           lower=r['lower'], upper=r['upper'], actual_loss=b['row']['actual_loss'],
                           conflict_freeze=b['row']['conflict_freeze'])
                failures.append(row)
    write_csv('coverage_failures.csv', failures)
    rows = [b['row'] for b in bindings if b['row']['status'] == 'ok']
    target_methods = defaultdict(dict)
    for r in rows: target_methods[r['target']][r['method']] = r
    contrasts = []
    for target, methods in target_methods.items():
        reference = min((methods[m] for m in ('right', 'cover', 'dynamic_max_degree')),
                        key=lambda r: (r['complete_requests'], r['method']))
        for method in ('Ddegree', 'C', 'Eall_mask'):
            if method not in methods: continue
            r = methods[method]
            row = {k: r[k] for k in ('target', 'n', 'k', 'law', 'reuse', 'seed', 'method')}
            row.update(exact_requests=reference['complete_requests'], complete_requests=r['complete_requests'],
                       saving=reference['complete_requests']-r['complete_requests'],
                       saving_per_n=(reference['complete_requests']-r['complete_requests'])/r['n'],
                       actual_loss=r['actual_loss'], cpu_seconds=r['cpu_seconds'])
            contrasts.append(row)
    for axis in ('n', 'k', 'law', 'reuse'):
        for value in sorted({r[axis] for r in contrasts}):
            for method in ('Ddegree', 'C', 'Eall_mask'):
                subset = [r for r in contrasts if r[axis] == value and r['method'] == method]
                if not subset: continue
                axis_rows.append(dict(axis=axis, value=value, method=method, targets=len(subset),
                    wins=sum(r['saving'] > 0 for r in subset), ties=sum(r['saving'] == 0 for r in subset),
                    losses=sum(r['saving'] < 0 for r in subset), mean_saving=float(np.mean([r['saving'] for r in subset])),
                    mean_saving_per_n=float(np.mean([r['saving_per_n'] for r in subset])),
                    mean_loss=float(np.mean([r['actual_loss'] for r in subset])),
                    mean_cpu_seconds=float(np.mean([r['cpu_seconds'] for r in subset]))))
    write_csv('paired_axis_summary.csv', axis_rows)
    ed = []
    for target, methods in target_methods.items():
        if 'Eall_mask' not in methods: continue
        e, d = methods['Eall_mask'], methods['Ddegree']
        row = {k: e[k] for k in ('target', 'n', 'k', 'law', 'seed')}
        row.update(e_requests=e['complete_requests'], d_requests=d['complete_requests'],
                   e_request_saving_vs_d=d['complete_requests']-e['complete_requests'],
                   e_cpu_seconds=e['cpu_seconds'], d_cpu_seconds=d['cpu_seconds'],
                   e_loss=e['actual_loss'], d_loss=d['actual_loss'])
        ed.append(row)
    write_csv('e_vs_d_matched.csv', ed)
    reuse_rows = []
    paired_reuse = defaultdict(dict)
    for r in rows: paired_reuse[(r['n'], r['k'], r['law'], r['seed'], r['method'])][r['reuse']] = r
    for key, pair in paired_reuse.items():
        if len(pair) != 2: continue
        low, high = pair[0.], pair[.25]
        reuse_rows.append(dict(n=key[0], k=key[1], law=key[2], seed=key[3], method=key[4],
            request_change=high['complete_requests']-low['complete_requests'],
            loss_change=high['actual_loss']-low['actual_loss'], full_count_change=high['full_count']-low['full_count'],
            same_primary_and_labels=specs[low['target']]['truth'] == specs[high['target']]['truth'] and
                                    specs[low['target']]['primary'] == specs[high['target']]['primary']))
    write_csv('paired_reuse_contrasts.csv', reuse_rows)
    # Reprice the same fixed executed ledgers. No variants, masks, or outputs
    # are rerun or selected as a function of helper price.
    sensitivity, sensitivity_summary, break_even = [], [], []
    for target, methods in target_methods.items():
        exact = {m: methods[m] for m in ('right', 'cover', 'dynamic_max_degree')}
        exact['descriptive_minimum_executed_exact'] = min(exact.values(), key=lambda r: (r['oracle_requests'], r['method']))
        for method in ('Ddegree', 'C', 'Eall_mask'):
            if method not in methods: continue
            r = methods[method]
            for comparator, reference in exact.items():
                for eta in (0., .01, .1, 1.):
                    cost = r['oracle_requests'] + eta * r['helper_requests']
                    baseline = reference['oracle_requests']
                    row = {k: r[k] for k in ('target', 'n', 'k', 'law', 'reuse', 'seed', 'method')}
                    row.update(comparator=comparator, comparator_executed_method=reference['method'], helper_price_eta=eta,
                        oracle_requests=r['oracle_requests'], helper_requests=r['helper_requests'],
                        observed_oracle_synthetic_tokens=r['oracle_tokens'], observed_helper_synthetic_tokens=r['helper_tokens'],
                        oracle_unit_equivalent_cost=cost, token_weighted_unit_cost=2*cost,
                        exact_oracle_unit_cost=baseline, absolute_unit_saving=baseline-cost,
                        saving_fraction=(baseline-cost)/baseline if baseline else 0.,
                        actual_loss=r['actual_loss'])
                    sensitivity.append(row)
            if r['helper_requests']:
                ref = exact['descriptive_minimum_executed_exact']
                break_even.append(dict(target=target, method=method, oracle_requests=r['oracle_requests'],
                    helper_requests=r['helper_requests'], exact_requests=ref['oracle_requests'],
                    break_even_helper_eta=(ref['oracle_requests']-r['oracle_requests'])/r['helper_requests']))
    write_csv('helper_price_sensitivity.csv', sensitivity)
    for comparator in ('right', 'cover', 'dynamic_max_degree', 'descriptive_minimum_executed_exact'):
        for method in ('Ddegree', 'C', 'Eall_mask'):
            for eta in (0., .01, .1, 1.):
                subset = [r for r in sensitivity if r['comparator'] == comparator and r['method'] == method and r['helper_price_eta'] == eta]
                sensitivity_summary.append(dict(comparator=comparator, method=method, helper_price_eta=eta, targets=len(subset),
                    mean_oracle_requests=float(np.mean([r['oracle_requests'] for r in subset])),
                    mean_helper_requests=float(np.mean([r['helper_requests'] for r in subset])),
                    mean_oracle_unit_equivalent_cost=float(np.mean([r['oracle_unit_equivalent_cost'] for r in subset])),
                    mean_exact_unit_cost=float(np.mean([r['exact_oracle_unit_cost'] for r in subset])),
                    mean_absolute_unit_saving=float(np.mean([r['absolute_unit_saving'] for r in subset])),
                    mean_saving_fraction=float(np.mean([r['saving_fraction'] for r in subset])),
                    wins=sum(r['absolute_unit_saving'] > 1e-9 for r in subset),
                    ties=sum(abs(r['absolute_unit_saving']) <= 1e-9 for r in subset),
                    losses=sum(r['absolute_unit_saving'] < -1e-9 for r in subset)))
    write_csv('helper_price_summary.csv', sensitivity_summary)
    write_csv('helper_price_break_even.csv', break_even)
    failure_cells = sorted({(r['target'], r['method']) for r in failures})
    geometry = json.loads((HERE / 'geometry_validation.json').read_text(encoding='utf-8'))
    lines = ['# Interpretation and limits', '',
        f"The frozen roster completed all 594 policy cells with 594 actual SQL materializations and no execution failures. The 8 uncovered region records belong to {len(failure_cells)} policy cells; coverage_failures.csv identifies all of them. No actual loss exceeded 0.1. This observed result does not establish a confidence guarantee from 108 constructed targets.", '',
        'Oracle acquisition and helper acquisition have separate ledgers. C has mean 830.69 oracle requests and 896.00 helper requests. Its equal-unit total is 1726.69 request units. All three exact controls have zero helper requests and mean oracle counts 891.86 (right), 891.93 (cover), 891.65 (dynamic_max_degree). C loses to the descriptive best executed exact control on all 108 targets when every helper request has the same unit price as an oracle request. This result is driven by the frozen artificial helper charge and must not be generalized to structural acquisition itself. Oracle input/output and helper input/output each carry 1 synthetic token per request: two tokens per request, with no estimated live token costs.', '',
        'Ddegree improves equal-unit totals on 66 targets, ties 22, and loses 20 to the descriptive minimum of the three executed exact controls; its mean absolute saving is 35.25 oracle-request units. Eall_mask improves 36, ties 8, and loses 10 on its 54 eligible targets, with mean absolute saving 56.22 units. C has mean absolute excess 836.17 units at equal prices. These numbers are absolute request units, not percentages. Audits are included. The minimum comparator is posthoc descriptive and is not presented as an executable method.', '',
        'helper_price_sensitivity.csv and helper_price_summary.csv reprice the identical frozen executed ledgers at eta in {0,.01,.1,1}, defining cost=oracle_requests+eta*helper_requests in oracle-request units. Multiplying by 2 gives the equivalent synthetic token-weighted cost because each tape request carries two tokens. eta 0 is the derived oracle-only comparison; it does not erase recorded helper access or its CPU. Outputs, paid facts, masks, audits, CPU, and targets are unchanged. Each eta is reported against every fixed exact control and the descriptive minimum; no variant is rerun or chosen retrospectively.', '',
        '| C helper price eta | Targets | Mean oracle-unit cost | Mean absolute saving vs descriptive exact minimum | Win/tie/loss |',
        '|---:|---:|---:|---:|---:|']
    for r in sensitivity_summary:
        if r['method'] == 'C' and r['comparator'] == 'descriptive_minimum_executed_exact':
            lines.append(f"| {r['helper_price_eta']:.2f} | {r['targets']} | {r['mean_oracle_unit_equivalent_cost']:.2f} | {r['mean_absolute_unit_saving']:.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ['',
        f"On the same 54 rho 0 targets, Eall_mask uses fewer/tied/more complete requests than Ddegree in {sum(r['e_request_saving_vs_d']>0 for r in ed)}/{sum(r['e_request_saving_vs_d']==0 for r in ed)}/{sum(r['e_request_saving_vs_d']<0 for r in ed)} cases, with mean request saving {np.mean([r['e_request_saving_vs_d'] for r in ed]):.2f}. Its mean complete CPU is {np.mean([r['e_cpu_seconds'] for r in ed]):.4f}s versus Ddegree {np.mean([r['d_cpu_seconds'] for r in ed]):.4f}s. Snapshot mask optimality does not imply an end-to-end cost dominance theorem.", '',
        'The four metadata axes are varied explicitly. Only the reuse contrast holds both primary assignment and realized labels fixed within each N/K/law/seed combination. N, K, and law also change the composition induced by the required cycling positive rates. K 2 has no key at the 0.85 rate, and Zipf emphasizes early keys. Therefore marginal axis means are descriptive, and cannot attribute all differences solely to key granularity or fan-out. target_descriptors.csv reports realized prevalence, effective nonempty keys, incidence count, degree variation, and structural bag size to expose this limitation. Empty keys were retained; no target was removed.', '',
        'paired_axis_summary.csv reports complete-cost savings, normalized by N as well as absolute requests, for each axis. paired_reuse_contrasts.csv records the direct matched rho 0-to-.25 contrast. Eall_mask has no rho .25 arm because repeated identities violate its eligibility contract; its overall means must not be compared with 108-target means without matching.', '',
        'Geometry uses explicitly covered constructed constraints: 48 targets and 3168 cache/interval/mask cases, with exhaustive worlds for N 8 or 12. rho 0 is exactly sharp in all 792 cases. For rho .25/.5/1 the copied envelope has strict gaps in 68/151/156 of 792 cases, maximum gaps 20/40/49, and misses 4/4/13 Boolean certifications respectively. The envelope never understates the shared-identity Boolean maximum in this suite. These are geometric results, not statistical audits or random confidence experiments.', '',
        'CPU and elapsed times come from one local complete run per cell and include SQL setup, reference and returned materialization, and validation. They are not isolated benchmark repetitions or model inference latency. Local work may share host resources. The geometry supplement ran briefly concurrently; the principal cost comparison uses deterministic charged requests/tokens, while wall-time differences should not be treated as precise speedups.', '',
        'No source from a reviewer report was read. No existing algorithm, core driver, manuscript, or protocol was changed. FREEZE.json and frozen_sources/ pin the exact study sources before checks/policy execution. This analyzer is post-run descriptive work; its hash and creation timestamp are separately recorded in post_analysis_manifest.json.']
    (HERE / 'INTERPRETATION.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    classification='post-run descriptive analysis, no policy selection or reruns',
                    coverage_failure_records=len(failures), coverage_failure_cells=len(failure_cells),
                    helper_price_sensitivity_rows=len(sensitivity), helper_price_summary=sensitivity_summary,
                    e_vs_d=dict(wins=sum(r['e_request_saving_vs_d']>0 for r in ed),
                                ties=sum(r['e_request_saving_vs_d']==0 for r in ed),
                                losses=sum(r['e_request_saving_vs_d']<0 for r in ed),
                                mean_request_saving=float(np.mean([r['e_request_saving_vs_d'] for r in ed]))))
    (HERE / 'post_analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('created_utc', 'source_sha256', 'coverage_failure_records',
                                              'coverage_failure_cells', 'helper_price_sensitivity_rows', 'e_vs_d')}))


if __name__ == '__main__': main()
