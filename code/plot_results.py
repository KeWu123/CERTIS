"""Equal-size ICDE result figures and a provenance-backed two-column table."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / 'reproduction/figures'
FIG = ROOT / 'outputs/figures'
DATA = json.loads((NOTES / 'plot_data.json').read_text(encoding='utf-8'))
PALETTE_PATH = NOTES / 'paper-palette.yaml'
PALETTE = json.loads(PALETTE_PATH.read_text(encoding='utf-8'))
C = PALETTE['methods']
WIDTH, HEIGHT = 516 / 72.27, 2.55
RECORDS = []


def style():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8,
        'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'xtick.major.width': .8, 'ytick.major.width': .8,
        'text.color': PALETTE['semantic']['ink'],
        'axes.labelcolor': PALETTE['semantic']['ink'],
        'xtick.color': PALETTE['semantic']['ink'],
        'ytick.color': PALETTE['semantic']['ink'],
        'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#FFFFFF',
    })


def canvas():
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = [fig.add_axes([x, .27, .385, .54]) for x in (.085, .585)]
    for ax in axes:
        ax.grid(axis='y', color=PALETTE['semantic']['stroke'], lw=.8, alpha=.45)
        ax.set_axisbelow(True)
    return fig, axes


def export(fig, stem):
    FIG.mkdir(exist_ok=True, parents=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            clipped.append(obj.get_text())
    assert not clipped, f'Clipped figure labels: {clipped}'
    for ax in fig.axes:
        tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()
                      if label.get_visible() and label.get_text()]
        assert all(a.x1 + 1 < b.x0 for a, b in zip(tick_boxes, tick_boxes[1:])), 'Overlapping x tick labels'
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{stem}.{ext}', dpi=300, facecolor='#FFFFFF', bbox_inches=None)
    RECORDS.append({'stem': stem, 'size_inches': list(fig.get_size_inches()),
                    'axes_positions': [list(ax.get_position().bounds) for ax in fig.axes],
                    'palette_id': PALETTE['palette_id'],
                    'palette_sha256': hashlib.sha256(PALETTE_PATH.read_bytes()).hexdigest(),
                    'data_sha256': hashlib.sha256((NOTES / 'plot_data.json').read_bytes()).hexdigest(),
                    'clipped_text': clipped})
    plt.close(fig)


def mechanisms():
    fig, axes = canvas()
    rows = DATA['matched_cells']
    specs = [('adaptive_per_side_requests', 'Adaptive per-side', 'per_side', '--', 's'),
             ('separate_requests', 'Separate extrema + greedy', 'separate_extrema', '-.', 'D'),
             ('greedy_requests', 'Joint + greedy', 'joint_greedy', '--', '^'),
             ('exact_requests', 'Joint + DP reference', 'joint_exact', '-', 'o')]
    for key, label, role, ls, marker in specs:
        axes[0].step(sorted(r[key] for r in rows), np.arange(1, 49)/48,
                     where='post', color=C[role], ls=ls, marker=marker,
                     markersize=3.1, markevery=9, lw=1.25, label=label)
    axes[0].set(xlabel='Complete oracle requests per cell',
                ylabel='Fraction of matched cells', ylim=(0, 1.025),
                xlim=(2850, 5050), title='(a) Certificate policies (48 cells)')
    axes[0].set_xticks([3000, 4000, 5000])
    axes[0].set_yticks([0, .25, .5, .75, 1])
    axes[0].legend(loc='upper left', borderaxespad=.25, labelspacing=.28, handlelength=2.3)
    labels = ['Per-key\nbudget', 'Frozen mask\n+ pruning', 'Separate\nextrema']
    keys = ['keywise_degree', 'fixed_mask_pruning_degree', 'separate_extrema_degree']
    values = []
    for key in keys:
        a = DATA['mechanisms'][key]['original_vs_control']
        values.append(100*a['pooled_saving_fraction'])
    bars = axes[1].barh(range(3), values,
                        color=[C['joint_exact'], C['degree_prefix'], C['per_side']], height=.55)
    for bar, key, value in zip(bars, keys, values):
        a = DATA['mechanisms'][key]['original_vs_control']
        axes[1].text(value+.22, bar.get_y()+bar.get_height()/2,
                     f'{value:.2f}%\n{a["wins"]}/{a["ties"]}/{a["losses"]}',
                     va='center', fontsize=8, linespacing=1.1)
    axes[1].set_yticks(range(3), labels)
    axes[1].set(xlim=(0, 15.8), xlabel='Request saving over matched control (%)',
                title='(b) Feedback ablations (96 cells)')
    axes[1].set_xticks([0, 5, 10, 15])
    axes[1].invert_yaxis()
    export(fig, 'fig_mechanisms')


def native():
    fig, axes = canvas()
    rows = DATA['strengthening']['native_rows']
    pairs = [('sent_positive', 'sent_negative'), ('approval', 'disapproval'),
             ('joy', 'sadness'), ('optimism', 'fear'), ('love', 'disgust'), ('gratitude', 'disappointment')]
    labels = ['Pos/neg', 'Appr/dis', 'Joy/sad', 'Opt/fear', 'Love/dis', 'Grat/dis']
    exact = {r['case_id']: r for r in rows if r['method'] == 'batched_right'}
    methods = [('adaptive_continuous', 'Adaptive per-side', 'per_side', 's', -.13),
               ('joint_greedy', 'Joint greedy (CERTIS)', 'joint_greedy', 'o', .13)]
    all_savings = [100*(1-r['requests']/exact[r['case_id']]['requests']) for r in rows]
    lo = 5*np.floor(min(all_savings)/5)-2
    hi = 5*np.ceil(max(all_savings)/5)+2
    handles = []
    for ax, suffix, title in zip(axes, ('subreddit', 'link_id'),
                                 ('(a) Subreddit keys', '(b) Thread keys')):
        cases = [f'goemotions_native__{a}__{b}__{suffix}' for a, b in pairs]
        for method, label, role, marker, shift in methods:
            vals = [[100*(1-r['requests']/exact[c]['requests'])
                     for r in rows if r['case_id'] == c and r['method'] == method] for c in cases]
            assert all(len(v) == 6 for v in vals)
            mean = np.array([np.mean(v) for v in vals])
            low = np.array([min(v) for v in vals])
            high = np.array([max(v) for v in vals])
            artist = ax.errorbar(np.arange(6)+shift, mean,
                                 yerr=np.vstack([mean-low, high-mean]),
                                 fmt=marker, color=C[role], markersize=4,
                                 elinewidth=.9, capsize=2, label=label)
            if ax is axes[0]:
                handles.append(artist)
        ax.axhline(0, color=C['exact_right'], lw=1, ls='--', zorder=1)
        sizes = [exact[c]['full_count'] for c in cases]
        size_labels = [f'{t/1000:.1f}k' if t >= 1000 else str(t) for t in sizes]
        ax.set_xticks(range(6), [f'{label}\nT={t}' for label, t in zip(labels, size_labels)])
        ax.set(xlim=(-.55, 5.55), ylim=(lo, hi), title=title,
               xlabel='Predicate pair; full bag size T',
               ylabel='Saving vs exact right-first (%)')
    fig.legend(handles, [s[1] for s in methods], loc='upper center',
               bbox_to_anchor=(.53, 1.01), ncol=2, columnspacing=1.6, handletextpad=.45)
    export(fig, 'fig_native')


def price_diagnostic():
    fig, axes = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=.1, right=.97, bottom=.25, top=.88)
    for method, label, role, marker in [
        ('direct_feedback_audit_degree', 'Degree prefix', 'degree_prefix', 's'),
        ('count_allocation_feedback_audit_degree', 'All-mask', 'joint_exact', 'o'),
        ('cut_feedback_audit', 'Paid cut', 'paid_cut', '^')]:
        for budget, ls in [(125, '-'), (500, '--')]:
            points = sorted([r for r in DATA['price_plot'] if r['method'] == method and r['budget'] == budget], key=lambda r:r['helper_to_oracle_token_price'])
            axes.plot([p['helper_to_oracle_token_price'] for p in points],
                      [p['saving_percent'] for p in points],
                      color=C[role], marker=marker, ls=ls, ms=3.5,
                      lw=1.2, label=f'{label}, audit {budget}')
    axes.set_xscale('symlog', linthresh=.01)
    axes.set_xticks([0, .01, .05, .1, 1], ['0', '.01', '.05', '.1', '1'])
    axes.axhline(0, color=C['exact_right'], lw=.9)
    axes.set(xlabel='Helper / oracle token price ratio (symlog)',
             ylabel='Modeled token-cost saving (%)', title='Native token-cost sensitivity vs exact right-first')
    axes.legend(loc='lower left', ncol=2)
    export(fig, 'diagnostic_token_prices')


def wide_table():
    lines = [r'\begin{table*}[t]', r'\centering',
        r'\caption{Complete oracle requests: greedy CERTIS versus seven exact controls on the original constructed and native matrices. CERTIS targets 90\% bag recall; exact execution returns the full bag.}',
        r'\label{tab:execution_overview}', r'\label{tab:extension_named_exact}', r'\label{tab:native_exact}',
        r'\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',
        r'\toprule',
        r'& & \multicolumn{2}{c}{\textbf{Audit 125/side}} & \multicolumn{2}{c}{\textbf{Audit 500/side}} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
        r'\textbf{Exact reference} & \textbf{Mean exact requests} & \textbf{Saving (\%)} & \textbf{W/T/L} & \textbf{Saving (\%)} & \textbf{W/T/L} \\',
        r'\midrule']
    for group in DATA['table_groups']:
        if group['cohort'] == 'constructed':
            title = r'\textbf{A. Constructed key-disjoint targets} (16 queries; 48 pairs/budget; 5,000 identities/query)'
        else:
            title = r'\textbf{B. Native discussion relationships} (eight queries; 24 pairs/budget; 9,996 identities/query)'
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{6}{l}{' + title + r'} \\')
        lines.append(r'\addlinespace[2pt]')
        for row in group['rows']:
            cells = [row['label'], f'{row["exact_mean_requests"]:,.1f}']
            for c in row['comparisons']:
                cells += [f'{100*c["pooled_saving_fraction"]:.2f}', f'{c["wins"]}/{c["ties"]}/{c["losses"]}']
            lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}',
              r'\par\smallskip\raggedright \textit{Notes.} CERTIS pays audit plus execution. Saving is $100(1-C_{\rm CERTIS}/C_{\rm exact})$ on pooled requests; negative values mean higher cost. '
              r'W/T/L counts fewer/equal/more requests. Each exact execution is paired with three CERTIS seeds; cohorts are summarized separately. '
              r'Native helper-greedy and negative-first controls additionally purchase 9,996 helper scores/query, excluded from these oracle-request savings.',
              r'\end{table*}']
    (FIG / 'execution_overview_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    assert (NOTES / 'figure-contracts.yaml').exists()
    assert PALETTE['status'] == 'frozen'
    style()
    mechanisms()
    native()
    price_diagnostic()
    wide_table()
    (FIG / 'figure_manifest.json').write_text(json.dumps(RECORDS, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('Exported two equal-size main figures, one price diagnostic, and the two-column result table.')
