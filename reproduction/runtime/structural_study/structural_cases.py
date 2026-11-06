"""Frozen exploratory structural generators; truth is harness-only input."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent / 'external_study'
sys.path.insert(0, str(EXTERNAL))
from common import make_tiny_case

SEEDS = (101, 202, 303)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def specification(n, k, law, reuse, seed, *, geometry=False):
    """Balanced sides; exact largest-remainder sizes; independently permuted IDs.

    Primary assignments and labels are paired across the reuse axis. Secondary
    membership is exactly floor(reuse * side size) on each side. Label generation
    has its own RNG, and never uses membership additions or any policy output.
    """
    assert n % 2 == 0 and k >= 2 and law in ('uniform', 'zipf1.5')
    m = n // 2
    law_id = int(law != 'uniform')
    weights = np.ones(k) if not law_id else np.arange(1, k + 1, dtype=float) ** -1.5
    weights /= weights.sum()
    raw = weights * m
    sizes = np.floor(raw).astype(int)
    for j in np.lexsort((np.arange(k), -(raw - sizes)))[:m - sizes.sum()]:
        sizes[j] += 1
    primary = []
    for side in (0, 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, k, law_id, side, 11]))
        values = np.repeat(np.arange(k), sizes)
        rng.shuffle(values)
        primary.extend(map(int, values))
    secondary = [-1] * n
    for side in (0, 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, k, law_id, side, 17]))
        selected = rng.permutation(m)[:int(m * reuse)] + side * m
        for j in selected:
            choice = int(rng.integers(k - 1))
            secondary[int(j)] = choice + int(choice >= primary[int(j)])
    label_rng = np.random.default_rng(np.random.SeedSequence([seed, n, k, law_id, 29]))
    rates = np.asarray((.15, .5, .85))
    truth = (label_rng.random(n) < rates[np.asarray(primary) % 3]).astype(int).tolist()
    left, right = [], []
    for key in range(k):
        left.append([j for j in range(m) if primary[j] == key or secondary[j] == key])
        right.append([j for j in range(m, n) if primary[j] == key or secondary[j] == key])
    name = f"{'geometry' if geometry else 'structural'}_n{n}_k{k}_{law}_r{str(reuse).replace('.', 'p')}_s{seed}"
    spec = dict(target=name, n=n, k=k, law=law, reuse=reuse, seed=seed,
                side=[0] * m + [1] * m, primary=primary, secondary=secondary,
                key_left=left, key_right=right, truth=truth,
                scores=[.5] * n, token_contract='oracle input=1 output=1; helper input=1 output=1',
                primary_counts_per_side=sizes.tolist(), positive_rates=[.15, .5, .85],
                rng='numpy.default_rng SeedSequence; primary tag11, secondary tag17, labels tag29')
    spec['input_sha256'] = digest(spec)
    return spec


def build(spec):
    case = make_tiny_case(side=spec['side'], key_left=spec['key_left'],
                         key_right=spec['key_right'], truth=spec['truth'],
                         scores=spec['scores'], name=spec['target'])
    # Stable content version overrides the factory's process-local tape key.
    case.metadata.update(target_version=spec['input_sha256'],
                         oracle_contract_version='structural-fixed-synthetic-v1',
                         input_sha256=spec['input_sha256'],
                         helper_scope='artificial uninformative score .5; all accesses charged unit input/output tokens')
    return case


def full_grid():
    return [specification(n, k, law, reuse, seed)
            for n in (128, 512, 2048) for k in (2, 8, 32)
            for law in ('uniform', 'zipf1.5') for reuse in (0., .25) for seed in SEEDS]


def geometry_grid():
    return [specification(n, k, 'uniform', reuse, seed, geometry=True)
            for n in (8, 12) for k in (2, 4)
            for reuse in (0., .25, .5, 1.) for seed in SEEDS]
