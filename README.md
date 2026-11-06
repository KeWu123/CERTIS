# CERTIS

Research code for recall certification of two-sided semantic equality joins under bag semantics.
The artifact includes the integer certificate, count-space planners, charged execution harness,
fixed oracle tapes, SQL validators, and the recorded comparisons used in the experiments.

The complete paper evidence map is in [PAPER_RESULTS.md](PAPER_RESULTS.md).
The extended artifact includes all 1,376 primary executions, all 594 structural
executions, 3,168 Boolean geometry cases, full matched-comparison entry points,
planner scaling, independent audit calibration and local closed-loop inference.
The native extension includes all 16 released-category DBpedia queries (224
complete executions), alongside the 12 natural discussion queries.

## Quick start

Use Python 3.13 and run commands from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements-recorded.txt
python reproduce.py verify
python reproduce.py check
python reproduce.py smoke
python reproduce.py analyze
python reproduce.py plot
```

`requirements.txt` specifies compatible dependency ranges; `requirements-recorded.txt`
records the environment used for this release. No model download, GPU, or API key is needed
for the certificate checks, tape-based execution, SQL validation, aggregation, or figures.
The smoke run executes a constructed query and both natural key variants, compares ordered
paid ledgers and masks with the frozen records, and independently validates the returned SQL bags.

## Code layout

| Location | Contents |
| --- | --- |
| `code/bilinear_certificate.py` | Integer count domains and exact key support |
| `code/count_mask_planner.py` | Four-pair frontier, local profiles, exact fixed-cache DP |
| `reproduction/runtime/greedy_count_profile/` | Practical greedy planner and execution adapters |
| `reproduction/runtime/external_study/` | Charged oracle interface, audits, exact controls, SQL validators |
| `reproduction/study/` | Per-side controls, natural workloads, memoization, timing, and aggregation |
| `reproduction/runtime/external_inputs/` | Prepared immutable oracle/helper arrays |
| `reproduction/runtime/candidate_systems/external/semviews/` | Pinned upstream Parquet files, predicates, and license |
| `reproduction/figures/` | Frozen plot inputs and style specifications |
| `reproduction/archive/` | Complete primary, structural, original native and mechanism archives |
| `reproduction/completion/` | Full-matrix entry points, scale/geometry/calibration and live inference |
| `outputs/` | Newly generated execution records and check reports; ignored by Git |

## Reproduce the comparisons

```bash
# Four per-side controls on the constructed targets.
python reproduce.py run --kind constructed --output outputs/constructed

# Greedy, adaptive per-side, and batched exact execution on natural queries.
python reproduce.py run --kind native --output outputs/native

# Optional query-local memoization check against the supplied natural traces.
python reproduce.py run-cached --output outputs/cached

# Reaggregate the supplied frozen records and regenerate figures/tables.
python reproduce.py analyze
python reproduce.py plot
```

Add `--pilot` to either `run` command for a smaller execution. A resumed run must have the
same source/input binding; use a new output directory after changing code or dependencies.
Current fixed-continuous execution uses the full product-recall budget. The stored
`constructed_v1` records also retain the earlier attained-recall comparator; aggregation
uses `fixed_continuous_v2` for the final fixed-continuous comparison.

The main 48-cell, audit-500 comparison records 232,371 requests for adaptive per-side
certification, 198,142 for joint greedy, and 197,563 for joint DP. Greedy saves 14.73% against
the adaptive per-side control at the same 90% bag-recall target. DP supplies the fixed-cache
discard optimum. Natural results include tasks where batched exact execution is cheaper.
On the full 16-query native DBpedia roster, greedy reduces requests by 12.65%
against adaptive per-side certification in all 96 matched cells. Against exact
right-first, coarse category keys save requests while finer keys regress;
all groups are retained. The separate local closed-loop study is 1.29% slower
than exact in pooled time across two passes, despite improving 1.18% over
per-side execution. These local measurements do not establish universal gains.

To verify all historical matrices and the seven matched comparison totals, run
`python reproduce.py verify-paper`. Full fresh execution commands and their exact
source-record directories are listed in [PAPER_RESULTS.md](PAPER_RESULTS.md).
For example:

```bash
python reproduce.py run-matched --pilot --output outputs/matched_pilot
python reproduce.py run-primary --output outputs/primary
python reproduce.py run-structural --output outputs/structural
python reproduce.py run-scaling --output outputs/scaling
python reproduce.py run-certificate-grid --output outputs/certificate_grid
python reproduce.py run-calibration --output outputs/calibration
```

Recorded timings describe their original measurement environment. The local Qwen batch
scenario measures fixed acquisition traces; recall validation uses the Llama-3.3-70B tapes.
To repeat local inference with your own runtime and model files:

```bash
python reproduction/study/local_batch_benchmark.py --server PATH_TO_LLAMA_SERVER --model PATH_TO_GGUF --output ../../outputs/local_timing
```

The separate closed-loop experiment calls the model for every paid acquisition,
uses returned labels to update execution, and checks them against its own fixed
Qwen reference. Its hash-selected 2,048-comment target and one-slot inference
configuration are declared in `reproduction/completion/RUNTIME_ADDENDUM.md`.
One-slot cost results must not be pooled with the earlier concurrent fixed-trace
scenario or the archived 70B oracle-quality results.

```bash
python reproduce.py run-closed-loop --server PATH_TO_LLAMA_SERVER --model PATH_TO_GGUF --output outputs/closed_loop
```

## Exact piecewise profile extension

The optional profile builder preserves every profile value, left-first witness,
and greedy allocation decision. It partitions each fixed-count split into at
most five quadratic-envelope pieces and tests exact integer root neighbours.
A guarded NumPy implementation uses integer `isqrt`; unsafe ranges use the
arbitrary-precision reference. The 1,024-left-unknown short-loop cutoff is fixed.

On 14 paired cold-cache configurations, the largest heterogeneous/skewed states
reduce profile-plus-heap time from 4.68/10.49 s to 0.589/0.637 s at N=131,072
and K=8. The imbalanced case increases from 0.518 to 0.592 s. Initial pure
Python measurements, including regressions, are retained. These are planner
measurements, separate from the original live-inference cost tables.

All 32 native DBpedia query/budget cells (seed 101; 64 fresh executions) have
identical old/new ordered ledgers, masks, facts and SQL bag counts. Their 4,147
distinct profile states were compared entry by entry. The original certificate
and default archived execution paths remain available.

```bash
python reproduce.py validate-fast-profile
python reproduce.py verify-fast-profile
python reproduce.py benchmark-fast-profile --output outputs/profile_timing
python reproduce.py replay-fast-profile --output outputs/profile_equivalence
```

`reproduction/fast_profile/` contains the argument, protocols, validation,
source versions and both timing series. Fresh timing depends on the local
machine. Replays include equality-check overhead and are not latency benchmarks.

## Data and integrity

Upstream inputs are pinned to [SemViews](https://github.com/semviews/lotus-semviews/tree/b102ced758428a47191a2a83f1890e44bb715e41)
commit `b102ced758428a47191a2a83f1890e44bb715e41`. Third-party license and data details remain
with that subset; see [THIRD_PARTY.md](THIRD_PARTY.md).

`artifact_manifest.json` binds the distributed files. Local filesystem metadata in exported
records is replaced with repository-relative paths. Each exported trace retains the digest
of its source record as `source_record_sha256`; summary bindings refer to the exported bytes.
Numerical outcomes, oracle identities, acquired labels/tokens, masks, audit regions, and
batch schedules are preserved. Original source identifiers describe the recorded experiment;
new runs bind the current release sources and inputs in their own manifests.
