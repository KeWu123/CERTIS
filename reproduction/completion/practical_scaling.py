"""Measure cold query-local memoization, including equal states across keys."""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import subprocess
import sys
import time
import tracemalloc

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import count_experiments as base


def worker(spec):
    keys=base.states(spec['n'],spec['k'],spec['law'])
    cache=lru_cache(maxsize=65536)(base.exact._numpy_profile)
    if spec['trace_memory']:tracemalloc.start()
    started,cpu=time.perf_counter(),time.process_time()
    local=tuple(cache(a,b,1,10) for a,b in keys)
    built=time.perf_counter()
    plan=base.allocate_profiles(local)
    finished=time.perf_counter()
    cpu_seconds=time.process_time()-cpu
    peak=tracemalloc.get_traced_memory()[1] if spec['trace_memory'] else None
    if spec['trace_memory']:tracemalloc.stop()
    result=dict(status='pass',**spec,profile_seconds=built-started,heap_seconds=finished-built,
        total_seconds=finished-started,cpu_seconds=cpu_seconds,
        traced_peak_bytes=peak,resident_peak_bytes=base.resident_peak(),
        cache=cache.cache_info()._asdict(),discarded=plan.discarded,support=plan.support,
        max_block=max(s.unknown for pair in keys for s in pair))
    print(json.dumps(result),flush=True)


def run(out):
    expected=json.loads((out/'scaling.json').read_text())['rows']
    expected={(r['n'],r['k'],r['law']):(r['discarded'],r['support']) for r in expected if r['status']=='pass'}
    binding=dict(driver_sha256=base.study.sha(__file__),kernel_driver_sha256=base.study.sha(base.__file__),
        addendum_sha256=base.study.sha(HERE/'CACHED_SCALING_ADDENDUM.md'))
    path=out/'practical_scaling_binding.json'
    if path.exists():assert json.loads(path.read_text())==binding
    else:base.study.save(path,binding)
    rows=[]
    for n in (1024,4096,16384,65536,131072):
        for k in (8,128):
            for law in ('uniform','zipf1.5'):
                for repetition in range(4):
                    spec=dict(n=n,k=k,law=law,repetition=repetition,trace_memory=repetition==3)
                    path=out/'practical_scaling_raw'/f'n{n}_k{k}_{law}_r{repetition}.json'
                    if path.exists():row=json.loads(path.read_text())
                    else:
                        try:
                            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',
                                json.dumps(spec)],capture_output=True,text=True,timeout=120)
                            if result.returncode:raise RuntimeError(result.stderr[-2000:])
                            row=json.loads(result.stdout)
                        except subprocess.TimeoutExpired:
                            row=dict(status='censored',deadline_seconds=120,**spec)
                        base.study.save(path,row)
                    if row['status']=='pass':
                        assert (row['discarded'],row['support'])==expected[n,k,law]
                    rows.append(row)
                    print(json.dumps(dict(stage='cold_memoized',**{key:row.get(key) for key in
                        ('n','k','law','repetition','status','total_seconds')})),flush=True)
    base.study.save(out/'practical_scaling.json',dict(status='pass',rows=rows,cells=20,
        timing_repetitions=3,memory_passes=1,source_binding=binding,
        cold_scope='Empty per-query exact-state caches; identical key states may hit within the cold query.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=base.ROOT/'outputs/count_completion_20261106')
    p.add_argument('--worker')
    args=p.parse_args()
    if args.worker:worker(json.loads(args.worker))
    else:run(args.output)
