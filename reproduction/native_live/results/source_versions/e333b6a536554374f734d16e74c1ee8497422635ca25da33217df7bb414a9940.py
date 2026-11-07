"""Timer-only revision of the isolated policy adapter before full-grid freeze.

The planner and original execution harness bytes are unchanged. Timers are
recorded around each planner invocation, without changing its inputs/outputs.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

from greedy_planner import plan_counts
from policy_adapter import HARNESS_SHA256

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT/'reproduction/runtime/external_study'
sys.path.insert(0, str(EXTERNAL))
path = EXTERNAL/'count_allocation_arms.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == HARNESS_SHA256
spec = importlib.util.spec_from_file_location('_private_greedy_count_execution_v2', path)
_execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_execution)
_observations = []


def _timed_plan_counts(*args, **kwargs):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        answer = plan_counts(*args, **kwargs)
    except Exception as exc:
        _observations.append(dict(wall_seconds=time.perf_counter()-wall,
                                  process_cpu_seconds=time.process_time()-cpu,
                                  status='exception', error_type=type(exc).__name__))
        raise
    _observations.append(dict(wall_seconds=time.perf_counter()-wall,
                              process_cpu_seconds=time.process_time()-cpu,
                              status='none' if answer is None else 'plan',
                              chosen_incident_unknown_discard_count=None if answer is None else answer.discarded,
                              scaled_support=None if answer is None else answer.support))
    return answer


_execution.plan_counts = _timed_plan_counts
CountEnvelope = _execution.CountEnvelope
eligible = _execution.eligible


def run_greedy_count(case, seed, audit_budget, *, audit=True, feedback=True, materialize=True):
    _observations.clear()
    result = _execution.run_count_allocation(case, seed, audit_budget, audit=audit,
                                            feedback=feedback, materialize=materialize)
    result['method'] = result['method'].replace('count_allocation_', 'greedy_count_profile_', 1)
    result['allocation_strategy'] = 'deterministic marginal-greedy over exact local count profiles'
    result['global_cardinality_optimality_claimed'] = False
    result['planner_observations'] = list(_observations)
    result['timings']['planner_process_cpu_seconds'] = sum(o['process_cpu_seconds'] for o in _observations)
    result['timings']['planner_wall_seconds'] = sum(o['wall_seconds'] for o in _observations)
    for event in result['events']:
        if 'optimal_counts' in event:
            event['chosen_counts'] = event.pop('optimal_counts')
        event['chosen_unknown_discard_count'] = event['discard_cardinality']
        event['chosen_incident_unknown_discard_count'] = (None if event['chosen_counts'] is None
            else sum(sum(pair) for pair in event['chosen_counts']))
    return result
