"""Preserve separately the envelope attempts and reached planner invocations.

Evidence may become inconsistent while constructing SideCounts, before the
planner is entered. That legitimate conflict-freeze attempt has no planner
timing observation. The v2 policy already preserves the mask and final None;
this revision makes the difference explicit in measurement metadata only.
"""
from policy_adapter_v2 import CountEnvelope, eligible
from policy_adapter_v2 import run_greedy_count as _run


def run_greedy_count(*args, **kwargs):
    result = _run(*args, **kwargs)
    reached = len(result['planner_observations'])
    attempts = result['projection_calls']
    if reached > attempts:
        raise AssertionError('More reached planner invocations than envelope attempts')
    result['planner_invocations'] = reached
    result['projection_attempts_without_planner'] = attempts-reached
    result['projection_attempts_without_planner_scope'] = 'Evidence construction can reject a feedback attempt before entering the planner'
    return result
