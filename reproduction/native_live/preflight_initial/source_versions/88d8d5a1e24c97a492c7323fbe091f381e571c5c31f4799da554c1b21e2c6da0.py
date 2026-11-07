"""Explicit isolated-module allocation-policy injection; production unchanged.

The pinned count_allocation_arms.py is executed in a new private module.
Only its plan_counts global binding is replaced. Its exact same CountEnvelope
and run_count_allocation therefore implement audit, rank, reconstruction,
monotone old masks, pruning, private Oracle cache and actual SQL evaluation.
Output names describe a heuristic count choice, never a global optimum.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

from greedy_planner import plan_counts

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / 'reproduction/runtime/external_study'
sys.path.insert(0, str(EXTERNAL))
HARNESS_SHA256 = '1a8c366927e55db6a911d734c53d63d6aa184e0a5a7323024b4788a1e07e749a'
path = EXTERNAL / 'count_allocation_arms.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != HARNESS_SHA256:
    raise AssertionError('Execution harness does not match pinned source')
spec = importlib.util.spec_from_file_location('_private_greedy_count_execution', path)
_execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_execution)
_execution.plan_counts = plan_counts

CountEnvelope = _execution.CountEnvelope
eligible = _execution.eligible


def run_greedy_count(case, seed, audit_budget, *, audit=True, feedback=True, materialize=True):
    result = _execution.run_count_allocation(case, seed, audit_budget, audit=audit,
                                            feedback=feedback, materialize=materialize)
    result['method'] = result['method'].replace('count_allocation_', 'greedy_count_profile_', 1)
    result['allocation_strategy'] = 'deterministic marginal-greedy over exact local count profiles'
    result['global_cardinality_optimality_claimed'] = False
    for event in result['events']:
        if 'optimal_counts' in event:
            event['chosen_counts'] = event.pop('optimal_counts')
    return result
