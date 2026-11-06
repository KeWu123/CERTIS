"""Controlled certificate geometry, cold planner scaling and audit calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'reproduction/study'),
               str(ROOT/'reproduction/runtime/external_study'),
               str(ROOT/'reproduction/runtime/greedy_count_profile'), str(ROOT/'code')]
import count_mask_planner as exact
from count_mask_planner import CountProfile, SideCounts
from greedy_planner import allocate_profiles
import common
import run_study as study
from direct_arms import downward_alpha
from exact_hypergeom import count_interval_exact


def sizes(n, k, law):
    weights = np.ones(k) if law == 'uniform' else np.arange(1, k+1, dtype=float)**-1.5
    raw = weights/weights.sum()*(n//2)
    result = np.floor(raw).astype(int)
    for index in np.lexsort((np.arange(k), -(raw-result)))[:n//2-int(result.sum())]:
        result[index] += 1
    return [int(x) for x in result]


def states(n, k, law, width=.3, seed=101):
    counts = sizes(n, k, law)
    order = np.random.default_rng(seed).permutation(k)
    pairs = []
    for i in order:
        m = counts[i]
        lo = max(0, math.floor(m*(.5-width/2)))
        hi = min(m, math.ceil(m*(.5+width/2)))
        pairs.append((SideCounts(lo, hi, 0, m), SideCounts(lo, hi, 0, m)))
    return tuple(pairs)


def profiles(keys, separate=False):
    if not separate:
        return tuple(exact._numpy_profile(a, b, 1, 10) for a, b in keys)
    result = []
    for a, b in keys:
        worst_loss = exact._numpy_profile(a, b, 0, 1)
        minimum_total = max(a.lower, a.positive)*max(b.lower, b.positive)
        result.append(CountProfile(worst_loss.forced,
            tuple(10*x-minimum_total for x in worst_loss.values), worst_loss.choices))
    return tuple(result)


def mechanism(out):
    rows = []
    for n in (128, 512, 2048):
        for k in (2, 8, 32):
            for law in ('uniform', 'zipf1.5'):
                for width in (0., .1, .3, .6, 1.):
                    for seed in (101, 202):
                        keys = states(n, k, law, width, seed)
                        joint = profiles(keys)
                        separate = profiles(keys, True)
                        assert all(all(a <= b for a, b in zip(j.values, s.values))
                                   for j, s in zip(joint, separate))
                        jp, sp = allocate_profiles(joint), allocate_profiles(separate)
                        assert jp is not None and sp is not None
                        # Re-evaluate both selected masks under both objectives.
                        j_under_s = sum(s.values[i] for s, i in zip(separate, jp.chosen_extra_counts))
                        s_under_j = sum(j.values[i] for j, i in zip(joint, sp.chosen_extra_counts))
                        assert s_under_j <= sp.support <= 0 and jp.support <= 0
                        row = dict(n=n, k=k, law=law, width=width, seed=seed,
                            max_block=max(s.unknown for pair in keys for s in pair),
                            joint_discard=jp.discarded, separate_discard=sp.discarded,
                            extra_discard=jp.discarded-sp.discarded,
                            joint_support=jp.support, separate_support=sp.support,
                            joint_mask_separate_support=j_under_s,
                            separate_mask_joint_support=s_under_j,
                            joint_mask_requires_coupling=j_under_s > 0,
                            count_states=[[asdict(s) for s in pair] for pair in keys])
                        rows.append(row)
    study.save(out/'mechanism.json', dict(status='pass', rows=rows, cells=len(rows),
        interpretation='Controlled fixed-cache count-state geometry, not measured query-request savings. '
        'Seeds permute key indices; equal-size cells can have identical count states.'))
    print(json.dumps(dict(stage='mechanism', cells=len(rows),
        strict_gain=sum(r['extra_discard']>0 for r in rows),
        fewer_discards=sum(r['extra_discard']<0 for r in rows),
        coupling_required=sum(r['joint_mask_requires_coupling'] for r in rows))), flush=True)


def resident_peak():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        assert ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def worker(spec):
    keys = states(spec['n'], spec['k'], spec['law'])
    if spec.get('trace_memory'):
        tracemalloc.start()
    started, cpu = time.perf_counter(), time.process_time()
    local = profiles(keys)
    built = time.perf_counter()
    plan = allocate_profiles(local)
    finished = time.perf_counter()
    process_seconds = time.process_time()-cpu
    traced = tracemalloc.get_traced_memory()[1] if spec.get('trace_memory') else None
    if spec.get('trace_memory'):
        tracemalloc.stop()
    peak_resident = resident_peak()
    existing = {(a,b,1,10):profile for (a,b),profile in zip(keys,local)}
    # Populate the lookup cache with already computed exact profiles. Timed
    # calls below are cache hits; no second cold build changes the memory peak.
    cache = lru_cache(maxsize=65536)(lambda a,b,p,q: existing[(a,b,p,q)])
    for a, b in keys:
        cache(a, b, 1, 10)
    warm = []
    for _ in range(3):
        t = time.perf_counter()
        reused = tuple(cache(a, b, 1, 10) for a, b in keys)
        cached_plan = allocate_profiles(reused)
        warm.append(time.perf_counter()-t)
        assert cached_plan.counts == plan.counts and cached_plan.support == plan.support
    dp = None
    if spec['n'] <= 4096 and not spec.get('trace_memory'):
        t = time.perf_counter()
        reference = exact.plan_counts(keys, Fraction(1,10))
        dp = dict(seconds=time.perf_counter()-t, discarded=reference.discarded)
        assert reference.discarded >= plan.discarded
    result = dict(status='pass', **spec, profile_seconds=built-started,
        heap_seconds=finished-built, total_seconds=finished-started, cpu_seconds=process_seconds,
        cold_profile_pair_evaluations=sum((a.unknown+1)*(b.unknown+1) for a,b in keys),
        max_block=max(s.unknown for pair in keys for s in pair), discarded=plan.discarded,
        support=plan.support, traced_peak_bytes=traced, resident_peak_bytes=peak_resident,
        warm_exact_state_cache_seconds=warm, dp=dp,
        count_states=[[asdict(s) for s in pair] for pair in keys])
    print(json.dumps(result), flush=True)


def scaling(out):
    rows = []
    for n in (1024, 4096, 16384, 65536, 131072):
        for k in (8, 128):
            for law in ('uniform', 'zipf1.5'):
                for repetition in range(4):
                    spec = dict(n=n, k=k, law=law, repetition=repetition, trace_memory=repetition==3)
                    path = out/'scaling_raw'/f'n{n}_k{k}_{law}_r{repetition}.json'
                    if path.exists():
                        row = json.loads(path.read_text(encoding='utf-8'))
                    else:
                        try:
                            run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                'worker', '--spec', json.dumps(spec)], capture_output=True,
                                text=True, encoding='utf-8', timeout=120)
                            if run.returncode:
                                raise RuntimeError(run.stderr[-2000:])
                            row = json.loads(run.stdout)
                        except subprocess.TimeoutExpired:
                            row = dict(status='censored', deadline_seconds=120, **spec)
                        study.save(path, row)
                    rows.append(row)
                    print(json.dumps(dict(stage='scaling', **{key:row.get(key) for key in
                        ('n','k','law','repetition','status','total_seconds','max_block')})), flush=True)
                    study.save(out/'scaling_progress.json', dict(completed=len(rows), expected=80))
    study.save(out/'scaling.json', dict(rows=rows, status='pass', expected=80,
        timing_repetitions=3, memory_passes=1, deadline_seconds=120,
        scope='Cold exact profile construction and greedy heap kernel; warm cache measured separately.'))


def calibration(out):
    rows = []
    for case in study.cases_for('native'):
        labels = common.Oracle(case)._validation_labels()
        domains = tuple(dict.fromkeys(ids for pair in zip(case.key_left, case.key_right) for ids in pair if ids))
        alpha = downward_alpha(Fraction(1,20)/max(1,len(domains)))
        side_populations = [np.flatnonzero(case.side==side) for side in (0,1)]
        cache = lru_cache(maxsize=65536)(count_interval_exact)
        for budget in (125,500):
            for seed in range(1001,1051):
                rng = np.random.default_rng(seed)
                sampled = set(int(j) for pop in side_populations for j in rng.choice(pop,
                    size=min(budget,len(pop)), replace=False))
                failures, conflicts = [], []
                for index, ids in enumerate(domains):
                    a = sum(j in sampled for j in ids)
                    b = sum(int(labels[j]) for j in ids if j in sampled)
                    lower, upper = cache(len(ids),a,b,alpha)
                    true = sum(int(labels[j]) for j in ids)
                    if not lower <= true <= upper:
                        failures.append(dict(region=index, population=len(ids), sample=a,
                            sample_positive=b, true_count=true, lower=lower, upper=upper))
                    # Full label revelation makes an incorrect interval empty.
                    if max(lower,true) > min(upper,true):
                        conflicts.append(index)
                rows.append(dict(case_id=case.case_id, seed=seed, budget=budget,
                    audit_event_id=f'{case.case_id}:s{seed}:b{budget}',
                    covered=not failures, failed_regions=failures,
                    inconsistent_at_full_revelation=bool(conflicts), regions=len(domains)))
        print(json.dumps(dict(stage='calibration', case_id=case.case_id,
            events=len(rows), failed_events=sum(not r['covered'] for r in rows))), flush=True)
        case.close_sql()
    study.save(out/'calibration.json', dict(status='pass', rows=rows, events=len(rows),
        seeds_per_query_budget=50, query_count=12, budgets=[125,500],
        failure_events=sum(not r['covered'] for r in rows),
        scope='One simultaneous audit event per query/seed/budget. Queries and budgets can share labels/draws; '
        'the pooled event count is descriptive, not 1,200 mutually independent trials.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['mechanism','scaling','calibration','worker'])
    p.add_argument('--output', type=Path, default=ROOT/'outputs/count_completion_20261106')
    p.add_argument('--spec')
    args = p.parse_args()
    if args.command == 'worker':
        worker(json.loads(args.spec))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        binding = dict(driver_sha256=study.sha(__file__), protocol_sha256=study.sha(HERE/'PROTOCOL.md'))
        path = args.output/(args.command+'_binding.json')
        if path.exists():
            assert json.loads(path.read_text()) == binding
        else:
            study.save(path,binding)
        globals()[args.command](args.output)
