"""Harness-only native relation restoration over immutable public model tapes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import common

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
PAIRS=(('sent_positive','anger'),('sent_positive','sadness'),
       ('joy','disapproval'),('admiration','annoyance'))


def load_native_cases():
    # Reuse the original exhaustive hash/input checks; no algorithm sees
    # these arrays directly. All accesses during execution use Oracle.
    original=common.load_cases()
    original_metadata=next(c.metadata for c in original if c.dataset=='goemotions')
    for c in original:c.close_sql()
    mapping_path=STUDY/'native_discussion/relation_metadata.parquet'
    manifest=json.loads((STUDY/'native_discussion/restoration_manifest.json').read_text(encoding='utf-8'))
    assert common.sha256(mapping_path)==manifest['mapping_sha256']
    mapping=pd.read_parquet(mapping_path).sort_values('row_id').reset_index(drop=True)
    assert len(mapping)==4998 and mapping.row_id.is_unique and mapping.comment_id.is_unique
    target_path=common.SOURCE/'data/datasets/goemotions.parquet'
    frame=pd.read_parquet(target_path).set_index('row_id')
    arrays=np.load(common.INPUTS/'goemotions_arrays.npz',allow_pickle=False)
    indices={str(r):i for i,r in enumerate(arrays['row_ids'])}
    rows=np.array([indices[r] for r in mapping.row_id],dtype=np.int64)
    pindex={str(p):i for i,p in enumerate(arrays['predicate_ids'])}
    n=len(mapping)
    result=[]
    for pl,pr in PAIRS:
        pl,pr='goemotions/'+pl,'goemotions/'+pr
        pi,pj=pindex[pl],pindex[pr]
        def select(field):return np.concatenate((arrays[field][pi,rows],arrays[field][pj,rows]))
        tape_key='native:'+hashlib.sha256((pl+pr+manifest['mapping_sha256']).encode()).hexdigest()
        tape=common._Tape(select('labels'),select('oracle_input_tokens'),select('oracle_output_tokens'),
                          select('helper_input_tokens'),select('helper_output_tokens'))
        for values in vars(tape).values():values.setflags(write=False)
        common._TAPES[tape_key]=tape
        for key_column in ('subreddit','link_id'):
            assert mapping[key_column].map(bool).all(),'Missing native relation key'
            groups=mapping.groupby(key_column,sort=True).indices
            keys=tuple(str(k) for k in groups)
            left=tuple(tuple(map(int,groups[k])) for k in groups)
            right=tuple(tuple(j+n for j in ids) for ids in left)
            ids=tuple(pl+'|'+r for r in mapping.row_id)+tuple(pr+'|'+r for r in mapping.row_id)
            version=hashlib.sha256(json.dumps((pl,pr,key_column,keys,manifest['mapping_sha256'])).encode()).hexdigest()
            metadata=dict(original_metadata)
            metadata.update(target_version='sha256:'+version,native_relation_key=key_column,
                mapping_sha256=manifest['mapping_sha256'],native_rows=n,native_unique_keys=len(keys),
                native_self_row_pairs_included=True,constructed_split=False)
            metadata['source_hashes']=dict(metadata['source_hashes'])
            metadata['source_hashes']['native_relation_metadata.parquet']=manifest['mapping_sha256']
            texts=tuple(str(frame.loc[r,'text']) for r in mapping.row_id)
            result.append(common.Case(dataset='goemotions_native',predicate_left=pl,predicate_right=pr,
                variant=key_column,ids=ids,side=np.array([0]*n+[1]*n),key_names=keys,
                key_left=left,key_right=right,scores=select('scores'),
                row_ids=tuple(mapping.row_id)*2,texts=texts*2,metadata=metadata,_tape_key=tape_key))
    return result


if __name__=='__main__':
    for case in load_native_cases():
        print(json.dumps(dict(case_id=case.case_id,n=case.n,keys=len(case.key_names),
             potential_pairs=sum(len(a)*len(b) for a,b in zip(case.key_left,case.key_right)),
             max_fanout=max(map(len,case.key_left))),ensure_ascii=False))
