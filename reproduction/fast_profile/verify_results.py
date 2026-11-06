"""Verify distributed profile benchmarks and native equivalence evidence."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def main():
    for version in (HERE/'source_versions').glob('*.py'):assert sha(version)==version.stem
    for cohort in ('python_reference','vectorized'):
        directory=HERE/'results'/cohort
        summary=read(directory/'summary.json')
        assert summary['status']=='pass' and len(summary['records'])==14
        for name,expected in summary['binding'].items():
            actual=ROOT/name
            if not actual.exists():actual=HERE/'source_versions'/(expected+'.py')
            assert actual.exists() and sha(actual)==expected,name
        for record in summary['records']:
            assert record['profile_and_allocation_equal']
            assert len(record['rows'])==6
            for backend in ('old','piecewise'):
                values=[r['total_seconds'] for r in record['rows'] if r['backend']==backend]
                assert len(values)==3 and statistics.median(values)==record['medians'][backend]
            assert abs(record['speedup']-record['medians']['old']/record['medians']['piecewise'])<1e-12
            for repetition in range(3):
                selected=[r for r in record['rows'] if r['repetition']==repetition]
                assert selected[0]['cache']==selected[1]['cache']
    directory=HERE/'results/equivalence'
    summary=read(directory/'summary.json');states=read(directory/'states.json')
    assert summary['matched_cells']==32 and summary['executions']==64
    assert len(states)==summary['unique_profile_states']==4147
    for key,state in states.items():assert digest(state)==key
    for row in summary['rows']:
        assert row['status']=='pass' and row['semantics_and_order_equal']
        path=ROOT/row['archive_path'];assert sha(path)==row['archive_sha256']
        reference=read(path)['result']
        for name in ('oracle_ledger','facts','discarded','regions','batches'):
            assert digest(reference[name])==row['signatures'][name],(row['case_id'],name)
        assert len(reference['oracle_ledger'])==row['requests']
        assert reference['output']['full_count']==row['full_count']
        assert reference['output']['returned_count']==row['returned_count']
        assert all(key in states for key in row['states'])
    output=ROOT/'outputs/fast_profile_evidence.json';output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='pass',benchmark_configurations=28,native_matched_cells=32,
        executions=64,unique_native_profile_states=len(states),archive_ledgers_rechecked=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
