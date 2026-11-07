"""Matched exact joint/separate profiles without modifying the published source."""
from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RELEASE = WORKSPACE / 'artifacts/CERTIS_release_20261104'
sys.path[:0] = [str(RELEASE/'reproduction/study'),
    str(RELEASE/'reproduction/runtime/external_study'),
    str(RELEASE/'reproduction/runtime/greedy_count_profile'), str(RELEASE/'code')]
import count_mask_planner as exact
from count_mask_planner import CountProfile
from greedy_planner import allocate_profiles, verify_profile_source
from piecewise_numpy_profile import numpy_piecewise_profile
import run_study as study


def build_profile(left, right, allowance, separate=False):
    """Same exact v3 builder; separate decouples loss and total extrema."""
    if not separate:
        return numpy_piecewise_profile(left, right, allowance)
    p, q = exact._fraction(allowance)
    loss = numpy_piecewise_profile(left, right, Fraction(0, 1))
    minimum_total = max(left.lower, left.positive)*max(right.lower, right.positive)
    return CountProfile(loss.forced, tuple(q*v-p*minimum_total for v in loss.values), loss.choices)


def mask_support(keys, counts, allowance, separate=False):
    """Evaluate these exact witnesses, not another profile's minimizing witness."""
    p, q = exact._fraction(allowance)
    total = 0
    for (a, b), (dl, dr) in zip(keys, counts):
        left = exact.RegionSide(a.lower, a.upper, a.positive, dl, a.unknown-dl)
        right = exact.RegionSide(b.lower, b.upper, b.positive, dr, b.unknown-dr)
        if separate:
            loss = exact.frontier_support(left, right, Fraction(0, 1))
            lower = max(a.lower, a.positive)*max(b.lower, b.positive)
            total += q*loss-p*lower
        else:
            total += exact.frontier_support(left, right, allowance)
    return total


def execute(case, seed, budget, method='joint', oracle_class=None, instrument=True):
    """Use existing executor; optional oracle_class(case) enables real inference.

    instrument=False builds only the selected profile family and avoids all
    dual-policy diagnostics. It is intended for actual cost measurements.
    """
    assert method in ('joint', 'separate')
    wrapper_started = time.perf_counter()
    verify_profile_source()
    spec = importlib.util.spec_from_file_location('_sharp_value_execution',
        study.EXTERNAL/'count_allocation_arms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cached_profile = lru_cache(maxsize=65536)(build_profile)
    support = lru_cache(maxsize=65536)(module.key_support)
    instances, observations, snapshots = [], [], []
    base_oracle = oracle_class or study.BatchOracle

    class Tracked(base_oracle):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    def plan(keys, allowance):
        started, cpu = time.perf_counter(), time.process_time()
        keys = tuple(tuple(pair) for pair in keys)
        is_separate = method == 'separate'
        local = tuple(cached_profile(a, b, allowance, is_separate) for a, b in keys)
        selected = allocate_profiles(local, backend='v3-exact-fast-'+method)
        if selected is None:
            return None
        joint_value = (mask_support(keys, selected.counts, allowance) if is_separate
            else selected.support)
        separate_value = (selected.support if is_separate else
            mask_support(keys, selected.counts, allowance, True) if instrument else None)
        if separate_value is not None:
            assert joint_value <= separate_value
        assert (separate_value if is_separate else joint_value) == selected.support <= 0
        observations.append(dict(wall_seconds=time.perf_counter()-started,
            cpu_seconds=time.process_time()-cpu, policy_support=selected.support,
            joint_support=joint_value, separate_support=separate_value))
        if instrument:
            alternatives = {}
            profiles = {}
            for kind in ('joint', 'separate'):
                profiles[kind] = (local if kind == method else tuple(
                    cached_profile(a, b, allowance, kind == 'separate') for a, b in keys))
                answer = selected if kind == method else allocate_profiles(profiles[kind])
                if answer is None:
                    alternatives[kind] = None
                else:
                    hj = mask_support(keys, answer.counts, allowance)
                    hs = mask_support(keys, answer.counts, allowance, True)
                    assert hj <= hs
                    assert (hj if kind == 'joint' else hs) == answer.support <= 0
                    alternatives[kind] = dict(discarded=answer.discarded,
                        counts=answer.counts, joint_support=hj, separate_support=hs,
                        requires_coupling=hj <= 0 < hs)
            # The minimal support at each cardinality also obeys dominance,
            # although the argmin witnesses can differ.
            assert all(all(j <= s for j, s in zip(jp.values, sp.values)) for jp, sp in
                zip(profiles['joint'], profiles['separate']))
            key_rows = []
            for index, (a, b) in enumerate(keys):
                key_rows.append(dict(index=index, key=case.key_names[index],
                    left=asdict(a), right=asdict(b),
                    original_left_size=len(case.key_left[index]),
                    original_right_size=len(case.key_right[index]),
                    interval_width_left=a.upper-a.lower,
                    interval_width_right=b.upper-b.lower,
                    unknown_imbalance=(max(a.unknown,b.unknown)/min(a.unknown,b.unknown)
                        if min(a.unknown,b.unknown) else None),
                    known_positives=a.positive+b.positive))
            snapshots.append(dict(index=len(snapshots), paid=len(instances[0].ledger),
                actual_method=method, keys=key_rows, alternatives=alternatives,
                same_evidence_discard_gap=(None if alternatives['separate'] is None else
                    alternatives['joint']['discarded']-alternatives['separate']['discarded'])))
        # Preserve the executor's original independent joint-kernel consistency
        # check, while retaining the policy's own support above.
        return replace(selected, support=joint_value)

    module.plan_counts, module.key_support, module.Oracle = plan, support, Tracked
    setup_seconds = time.perf_counter()-wrapper_started
    result = module.run_count_allocation(case, seed, budget, materialize=True)
    cleanup_started = time.perf_counter()
    result['method'] = 'global_'+method+'_greedy_fast'
    result['batches'] = instances[0].batches
    result['allocation_strategy'] = 'same deterministic marginal-greedy, query-local v3 exact profiles'
    result['global_cardinality_optimality_claimed'] = False
    result['policy_support_observations'] = observations
    if instrument:
        result['sharp_snapshots'] = snapshots
    result['cache_info'] = dict(profiles=cached_profile.cache_info()._asdict(),
        support=support.cache_info()._asdict())
    result['timings']['planner_wall_seconds'] = sum(x['wall_seconds'] for x in observations)
    result['timings']['planner_process_cpu_seconds'] = sum(x['cpu_seconds'] for x in observations)
    for event in result['events']:
        event['chosen_counts'] = event.pop('optimal_counts', None)
        event['chosen_unknown_discard_count'] = event['discard_cardinality']
        event['chosen_incident_unknown_discard_count'] = (None if event['chosen_counts'] is None
            else sum(sum(pair) for pair in event['chosen_counts']))
    cached_profile.cache_clear()
    support.cache_clear()
    cleanup_seconds = time.perf_counter()-cleanup_started
    result['timings']['cache_setup_seconds'] = setup_seconds
    result['timings']['cache_cleanup_seconds'] = cleanup_seconds
    result['timings']['algorithm_seconds'] += setup_seconds+cleanup_seconds
    result['diagnostic_instrumentation'] = instrument
    return result
