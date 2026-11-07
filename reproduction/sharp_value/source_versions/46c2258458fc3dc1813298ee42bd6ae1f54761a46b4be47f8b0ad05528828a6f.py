"""Complete native category-key workload over unchanged public oracle tapes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/study'),str(ROOT/'reproduction/runtime/external_study'),
    str(ROOT/'reproduction/runtime/greedy_count_profile'),str(ROOT/'code')]
import common
import run_study as study
from cached_greedy import execute


def cases_for():
    existing=common.load_cases()
    metadata=next(c.metadata for c in existing if c.dataset=='dbpedia')
    for c in existing:c.close_sql()
    frame=pd.read_parquet(common.SOURCE/'data/datasets/dbpedia.parquet').sort_values('row_id').reset_index(drop=True)
    arrays=np.load(common.INPUTS/'dbpedia_arrays.npz',allow_pickle=False)
    indices={str(r):i for i,r in enumerate(arrays['row_ids'])}
    rows=np.array([indices[str(r)] for r in frame.row_id],dtype=np.int64)
    pindex={str(p):i for i,p in enumerate(arrays['predicate_ids'])}
    n=len(frame);assert n==5000 and frame.row_id.is_unique
    cases=[]
    for pl,pr in common.DBPEDIA_PAIRS:
        pl,pr='dbpedia/'+pl,'dbpedia/'+pr
        def select(field):return np.concatenate((arrays[field][pindex[pl],rows],arrays[field][pindex[pr],rows]))
        key='native-dbpedia:'+pl+':'+pr
        tape=common._Tape(select('labels'),select('oracle_input_tokens'),select('oracle_output_tokens'),
            select('helper_input_tokens'),select('helper_output_tokens'))
        for v in vars(tape).values():v.setflags(write=False)
        common._TAPES[key]=tape
        for column in ('l1','l2'):
            assert frame[column].notna().all()
            groups=frame.groupby(column,sort=True).indices
            keys=tuple(str(k) for k in groups)
            left=tuple(tuple(map(int,groups[k])) for k in groups)
            right=tuple(tuple(j+n for j in ids) for ids in left)
            version=hashlib.sha256(json.dumps((pl,pr,column,keys,
                common.sha256(common.SOURCE/'data/datasets/dbpedia.parquet'))).encode()).hexdigest()
            meta=dict(metadata,target_version='sha256:'+version,native_relation_key=column,
                native_rows=n,native_unique_keys=len(keys),same_article_pairs_included=True,
                construction='Native equality join on the released DBpedia category fields',
                category_metadata_used_as_oracle_labels=False,constructed_split=False)
            cases.append(common.Case(dataset='dbpedia_native',predicate_left=pl,predicate_right=pr,
                variant=column,ids=tuple(pl+'|'+str(r) for r in frame.row_id)+tuple(pr+'|'+str(r) for r in frame.row_id),
                side=np.array([0]*n+[1]*n),key_names=keys,key_left=left,key_right=right,
                scores=select('scores'),row_ids=tuple(map(str,frame.row_id))*2,
                texts=tuple(map(str,frame.text))*2,metadata=meta,_tape_key=key))
    return cases


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cases=cases_for()
    if args.pilot:cases=cases[:1]
    files=[Path(__file__),HERE/'NATIVE_DBPEDIA_PROTOCOL.md',*study.source_paths(),
        ROOT/'reproduction/study/cached_greedy.py',ROOT/'code/count_mask_planner.py']
    binding=dict(sources={p.relative_to(ROOT).as_posix():study.sha(p) for p in files},
        dataset_sha256=study.sha(common.SOURCE/'data/datasets/dbpedia.parquet'),
        arrays_sha256=study.sha(common.INPUTS/'dbpedia_arrays.npz'),
        targets={c.case_id:c.target_version for c in cases},seeds=[101] if args.pilot else [101,202,303],
        budgets=[125,500],methods=['batched_left','batched_right','adaptive_continuous','joint_greedy_cached'])
    sid=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==dict(source_id=sid,**binding)
    else:study.save(out/'manifest.json',dict(source_id=sid,**binding))
    rows=[]
    for case in cases:
        jobs=[(m,0,0) for m in ('batched_left','batched_right')]
        jobs += [(m,s,b) for m in ('adaptive_continuous','joint_greedy_cached') for b in (125,500)
                 for s in ((101,) if args.pilot else (101,202,303))]
        for method,seed,budget in jobs:
            path=out/'raw'/f'{case.case_id}__{method}__s{seed}__b{budget}.json.gz'
            if path.exists():
                record=study.read(path);assert record['source_id']==sid
            else:
                case.close_sql();started=time.perf_counter()
                if method.startswith('batched'):result=study.batched_exact(case,method)
                elif method=='joint_greedy_cached':result=execute(case,seed,budget)
                else:result=study.marginal(case,seed,budget,method)
                output=result['output']
                # SQL setup/returned materialization are charged; full reference
                # construction and independent validation remain separate.
                sql=sum(output[k] for k in ('sql_preprocessing_seconds','fact_ingest_seconds',
                                            'returned_materialization_seconds'))
                cost=result['timings']['algorithm_seconds']+sql
                truth=common.Oracle(case)._validation_labels()
                coverage=all(r['lower']<=sum(int(truth[j]) for j in r['ids'])<=r['upper']
                             for r in result['regions'])
                assert not coverage or 10*output['lost_count']<=output['full_count']
                assert output['bag_subset']
                row=dict(case_id=case.case_id,key=case.variant,method=method,seed=seed,budget=budget,
                    requests=result['summary']['oracle_requests'],batches=len(result['batches']),
                    tokens_in=result['summary']['oracle_tokens_in'],tokens_out=result['summary']['oracle_tokens_out'],
                    local_seconds=cost,coverage=coverage,loss=output['loss'],full_count=output['full_count'],
                    returned_count=output['returned_count'],conflict=result.get('conflict_freeze',False),
                    outer_seconds=time.perf_counter()-started)
                record=dict(status='pass',source_id=sid,row=row,result=result,
                    target_version=case.target_version,metadata=case.metadata)
                study.save(path,record,True)
            rows.append(record['row'])
            study.save(out/'progress.json',dict(completed=len(rows),expected=len(cases)*(2+2*2*(1 if args.pilot else 3))))
            print(json.dumps(dict(completed=len(rows),**record['row'])),flush=True)
        case.close_sql()
    summary=dict(status='pass',rows=rows,executions=len(rows),queries=len(cases),source_id=sid)
    study.save(out/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/native_dbpedia_20261106')
    p.add_argument('--pilot',action='store_true')
    run(p.parse_args())
