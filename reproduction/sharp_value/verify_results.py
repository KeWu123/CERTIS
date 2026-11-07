"""Read-only integrity and headline recheck of the distributed sharp evidence."""
from pathlib import Path
import gzip, hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
summary=read(HERE/'results/summary.json')
assert summary['status']=='pass' and len(summary['rows'])==32
for row in summary['rows']:
    for kind,binding in row['raw'].items():
        p=HERE/'results'/binding['path']; assert sha(p)==binding['sha256']
        record=read(p); assert record['status']=='pass'
        assert len(record['result']['oracle_ledger'])==row[kind+'_requests']
    assert row['request_gap']==row['separate_requests']-row['joint_requests']
assert sum(r['joint_requests'] for r in summary['rows'])==268927
assert sum(r['separate_requests'] for r in summary['rows'])==269576
bound=read(HERE/'results/fractional_bound.json')
assert bound['cells']==32 and sum(r['proven_cardinality_advantage']>0 for r in bound['rows'])==15
for item in read(HERE/'source_index.json')['executed_sources']:
    p=ROOT/item['distributed_path']; assert sha(p)==item['sha256']
print('pass: 64 replay records, 32 cells, 15 strict initial-cache separations; no live inference')
