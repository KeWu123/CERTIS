"""Plot complete cold-planner measurements and controlled certificate geometry."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def run(data,output):
    scaling=json.loads((data/'practical_scaling.json').read_text())['rows']
    mechanism=json.loads((data/'mechanism_optima.json').read_text())['rows']
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'font.size':8.2,'axes.labelsize':8.2,'axes.titlesize':8.5,'xtick.labelsize':8.2,
        'ytick.labelsize':8.2,'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.25))
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.86,wspace=.34)
    colors={8:'#2D485E',128:'#55766F'}
    handles=[]
    for k in (8,128):
        for law,style,marker in [('uniform','-','o'),('zipf1.5','--','s')]:
            selected=[r for r in scaling if r['k']==k and r['law']==law and not r['trace_memory'] and r['status']=='pass']
            ns=sorted({r['n'] for r in selected})
            groups=[[r['total_seconds'] for r in selected if r['n']==n] for n in ns]
            median=[statistics.median(x) for x in groups]
            axes[0].fill_between(ns,[min(x) for x in groups],[max(x) for x in groups],color=colors[k],alpha=.08)
            line,=axes[0].plot(ns,median,style,marker=marker,markersize=3.5,
                color=colors[k],linewidth=1.1,label=f'K={k}, '+('uniform' if law=='uniform' else 'Zipf'))
            handles.append(line)
    axes[0].set_xscale('log',base=2);axes[0].set_yscale('log')
    axes[0].set_xticks([1024,16384,131072],['1k','16k','131k'])
    axes[0].set(xlabel='Unknown identities N',ylabel='Cold planning time (s)',title='(a) Empty query-local caches')
    axes[0].legend(handles=handles,loc='upper left',frameon=False,handlelength=2,borderaxespad=.2)
    widths=[0.,.1,.3,.6,1.]
    for law,color,marker in [('uniform','#2D485E','o'),('zipf1.5','#55766F','s')]:
        groups=[[100*r['exact_certificate_gain']/r['n'] for r in mechanism
            if r['law']==law and r['width']==w and r['seed']==101] for w in widths]
        axes[1].fill_between(widths,[min(g) for g in groups],[max(g) for g in groups],color=color,alpha=.12)
        axes[1].plot(widths,[statistics.median(g) for g in groups],marker=marker,
            color=color,linewidth=1.1,markersize=3.5,label='Uniform' if law=='uniform' else 'Zipf')
    axes[1].set(xlabel='Nominal interval width / block size',
        ylabel='Extra certified discards (% of N)',title='(b) Joint vs. separate exact optima')
    axes[1].set_xticks(widths,['0','.1','.3','.6','1'])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='upper right',frameon=False)
    for ax in axes:
        ax.grid(axis='y',color='#D8DCE0',linewidth=.45)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=2.5,width=.6)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'),dpi=220)
    plt.close(fig)
    sources={name:hashlib.sha256((data/name).read_bytes()).hexdigest()
        for name in ('practical_scaling.json','mechanism_optima.json')}
    output.with_suffix('.json').write_text(json.dumps(dict(sources=sources,
        canvas_inches=[7.05,2.25],minimum_font_points=8.2,
        statistics='Panel a: medians/ranges of three timed cold builds, excluding memory passes. '
        'Panel b: medians/ranges across all nine N/K settings per law and nominal width, seed101 key order. '
        'Second key permutation retained in raw data; bands are ranges, not confidence intervals. '
        'Count endpoints use outward integer rounding.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'reproduction/completion/results/counts')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/fig_completion')
    args=p.parse_args();run(args.data,args.output)
