"""Generate the paper table from the complete independently checked live matrix."""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
METHODS = ('batched_right', 'adaptive_continuous', 'separate', 'joint')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def count(value):
    return f'{value:,.0f}' if value == int(value) else f'{value:,.1f}'


def build(source, destination):
    report = read(source)
    assert report['status'] == 'pass' and report['validated_runs'] == 32
    assert len(report['queries']) == 4 and report['returned_first_sql']
    assert report['all_paid_model_labels_rechecked']
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Native Qwen closed-loop cost on all 5,000 descriptions per side. '
        r'Each entry is the mean of two timing passes (audit seeds 101/202 for approximate policies); '
        r'requests include every paid label. '
        r'Time includes actual inference, audit, planning, bookkeeping, and returned-bag SQL. '
        r'Loading, startup, warmup, and reference acquisition/verification are separate.}',
        r'\label{tab:native_live_cost}',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llr*{8}{r}@{}}',
        r'\toprule',
        r'\multicolumn{3}{c}{\textbf{Workload}} & \multicolumn{4}{c}{\textbf{Mean complete requests}} '
        r'& \multicolumn{4}{c}{\textbf{Mean complete time (s)}} \\',
        r'\cmidrule(lr){1-3}\cmidrule(lr){4-7}\cmidrule(l){8-11}',
        r'Predicate pair & Key & Full bag & Exact & Per-side & Separate & Joint '
        r'& Exact & Per-side & Separate & Joint \\',
        r'\midrule',
    ]
    emitted = []
    for query in report['queries']:
        _, left, right, key = query['case_id'].split('__')
        assert key == query['key']
        methods = query['methods']
        full_counts = {r['full_count'] for m in METHODS for r in methods[m]['runs']}
        assert len(full_counts) == 1
        values = [f'{left.title()}/{right.title()}', '$l_'+key[1]+'$', count(full_counts.pop())]
        values += [count(methods[m]['mean_requests']) for m in METHODS]
        values += [f"{methods[m]['mean_seconds']:.2f}" for m in METHODS]
        lines.append(' & '.join(values) + r' \\')
        emitted.append(dict(case_id=query['case_id'], cells=values))
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}', '']
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'native_live_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    provenance = dict(
        status='pass', source=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binding=report['binding'], table_rows=emitted,
        transformation='Per related query and method, arithmetic means of the two actual complete runs; '
            'format counts to integer or one decimal, times to two decimals. '
            'The full bag count is independently checked and invariant across all eight runs.')
    (destination / 'native_live_table_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='pass', queries=len(emitted), rows=report['validated_runs'],
                         source_sha256=provenance['source_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'native_live_analysis.json')
    parser.add_argument('--destination', type=Path, default=HERE / 'paper_evidence')
    args = parser.parse_args()
    build(args.source.resolve(), args.destination.resolve())
