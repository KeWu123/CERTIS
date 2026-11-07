"""Describe divergence of already executed policies; does not run an oracle."""
import argparse
import json
from pathlib import Path
from sharp_wrapper import HERE, study


def main(args):
    base=args.results.resolve()
    summary=json.loads((base/'summary.json').read_text())
    rows=[]
    for entry in summary['rows']:
        result={}
        for method in ('joint','separate'):
            binding=entry['raw'][method]
            assert study.sha(base/binding['path'])==binding['sha256']
            result[method]=study.read(base/binding['path'])['result']
        joint,separate=result['joint'],result['separate']
        jm,sm=set(joint['initial_discarded']),set(separate['initial_discarded'])
        jp={x['id']:x for x in joint['oracle_ledger']}
        sp={x['id']:x for x in separate['oracle_ledger']}
        jo,so=set(jp)-set(sp),set(sp)-set(jp)
        first=next((i for i,(x,y) in enumerate(zip(joint['oracle_ledger'],separate['oracle_ledger']))
            if x['id']!=y['id']),None)
        rows.append(dict(case_id=entry['case_id'],budget=entry['budget'],
            request_gap=entry['request_gap'],initial_joint_contains_separate=sm<=jm,
            initial_joint_only=len(jm-sm),initial_separate_only=len(sm-jm),
            joint_only_paid=len(jo),separate_only_paid=len(so),
            joint_only_positive=sum(jp[x]['label'] for x in jo),
            separate_only_positive=sum(sp[x]['label'] for x in so),
            first_ledger_divergence=first,joint_batches=len(joint['batches']),
            separate_batches=len(separate['batches'])))
    study.save(base/'execution_path_diagnostics.json',dict(status='pass',rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    main(parser.parse_args())
