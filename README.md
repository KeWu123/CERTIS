# CERTIS

Research code for recall certification of two-sided semantic equality joins under bag semantics.
The artifact includes the integer certificate, count-space planners, charged execution harness,
fixed oracle tapes, SQL validators, and the recorded comparisons used in the experiments.

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

Recorded timings describe their original measurement environment. The local Qwen batch
scenario measures fixed acquisition traces; recall validation uses the Llama-3.3-70B tapes.
To repeat local inference with your own runtime and model files:

```bash
python reproduction/study/local_batch_benchmark.py --server PATH_TO_LLAMA_SERVER --model PATH_TO_GGUF --output ../../outputs/local_timing
```

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
