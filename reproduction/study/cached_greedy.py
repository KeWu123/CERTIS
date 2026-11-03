"""Query-local memoization of pure exact kernels; identical allocation policy."""
from functools import lru_cache
import importlib.util
import sys
import time

from run_study import EXTERNAL, BatchOracle
import count_mask_planner as exact
from greedy_planner import allocate_profiles, verify_profile_source


def execute(case,seed,budget):
    wrapper_started=time.perf_counter()
    verify_profile_source()
    spec=importlib.util.spec_from_file_location('_query_local_cached_greedy_execution',EXTERNAL/'count_allocation_arms.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    numpy_profile=lru_cache(maxsize=65536)(exact._numpy_profile)
    python_profile=lru_cache(maxsize=65536)(exact._python_profile)
    support=lru_cache(maxsize=65536)(module.key_support)
    observations=[]; instances=[]

    def plan(keys,allowance):
        started=time.perf_counter(); cpu=time.process_time()
        keys=tuple(tuple(pair) for pair in keys)
        p,q=exact._fraction(allowance)
        safe=exact._int64_safe(keys,p,q)
        profiles=tuple(numpy_profile(a,b,p,q) if safe else python_profile(a,b,allowance) for a,b in keys)
        result=allocate_profiles(profiles,backend='numpy-int64' if safe else 'python-integer')
        observations.append(dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu))
        return result

    class Tracked(BatchOracle):
        def __init__(self,*a,**kw): super().__init__(*a,**kw); instances.append(self)

    module.plan_counts=plan; module.key_support=support; module.Oracle=Tracked
    setup_seconds=time.perf_counter()-wrapper_started
    result=module.run_count_allocation(case,seed,budget,materialize=True)
    cleanup_started=time.perf_counter()
    result['method']='joint_greedy_cached'
    result['batches']=instances[0].batches
    result['allocation_strategy']='same deterministic marginal-greedy with query-local exact-state memoization'
    result['global_cardinality_optimality_claimed']=False
    result['cache_info']={k:v._asdict() for k,v in [('numpy_profiles',numpy_profile.cache_info()),
        ('python_profiles',python_profile.cache_info()),('support',support.cache_info())]}
    result['timings']['planner_wall_seconds']=sum(x['wall_seconds'] for x in observations)
    result['timings']['planner_process_cpu_seconds']=sum(x['cpu_seconds'] for x in observations)
    for event in result['events']:
        if 'optimal_counts' in event: event['chosen_counts']=event.pop('optimal_counts')
        event['chosen_unknown_discard_count']=event['discard_cardinality']
        event['chosen_incident_unknown_discard_count']=(None if event['chosen_counts'] is None
            else sum(sum(pair) for pair in event['chosen_counts']))
    numpy_profile.cache_clear(); python_profile.cache_clear(); support.cache_clear()
    cleanup_seconds=time.perf_counter()-cleanup_started
    result['timings']['cache_setup_seconds']=setup_seconds
    result['timings']['cache_cleanup_seconds']=cleanup_seconds
    result['timings']['algorithm_seconds']+=setup_seconds+cleanup_seconds
    return result
