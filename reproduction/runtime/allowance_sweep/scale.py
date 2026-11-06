"""Scale experiment: exact all-mask planner wall-time vs N and key structure.

Confirms the O(K+N^2) complexity claim empirically and prices the two
backends (vectorised int64 vs arbitrary-precision Python-integer fallback).
N is capped at 100k because the count convolution is genuinely quadratic
(sums of profile widths ~ N, each convolved against a budget axis ~ N), so
the run maps the quadratic constant rather than pretending sub-quadratic.
"""
from __future__ import annotations

from fractions import Fraction
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2] / "code"
sys.path.insert(0, str(CODE))

from count_mask_planner import SideCounts, plan_counts


def build_instance(n, rng, heavy_tail=False):
    keys = []
    used = 0
    while used < n:
        if heavy_tail and rng.random() < 0.05:
            ls = int(rng.integers(15, 40)); rs = int(rng.integers(15, 40))
        else:
            ls = int(rng.integers(1, 4)); rs = int(rng.integers(1, 4))
        ls = min(ls, n - used); rs = min(rs, n - used)
        h_l = int(rng.integers(0, ls + 1)); h_r = int(rng.integers(0, rs + 1))
        hi_l = h_l + ls; hi_r = h_r + rs
        lo_l = max(0, hi_l - max(1, ls // 4)); lo_r = max(0, hi_r - max(1, rs // 4))
        keys.append((SideCounts(lo_l, hi_l, h_l, ls),
                     SideCounts(lo_r, hi_r, h_r, rs)))
        used += ls + rs
    return keys


def main():
    rng = np.random.default_rng(20261017)
    lam = Fraction(1, 10)
    rows = []
    for n in (10_000, 30_000, 100_000):
        for heavy in (False, True):
            keys = build_instance(n, rng, heavy_tail=heavy)
            k = len(keys)
            for backend in ("auto", "python"):
                t0 = time.perf_counter()
                ans = plan_counts(keys, lam, backend=backend)
                dt = time.perf_counter() - t0
                rows.append({"n": n, "heavy_tail": heavy, "keys": k,
                             "backend_requested": backend,
                             "backend_used": ans.backend if ans else None,
                             "seconds": dt, "discarded": ans.discarded if ans else None})
                print(f"n={n:7d} heavy={int(heavy)} keys={k:6d} {backend:7s}->"
                      f"{ans.backend:13s} {dt*1000:9.1f} ms  discard={ans.discarded}",
                      flush=True)
    out = HERE / "scale_results.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
