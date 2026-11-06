# Completion experiments

This exploratory extension is specified before its new results are inspected.
Existing paper results are known. No task is removed because of its outcome.

## Local closed-loop execution

- Corpus: all 4,998 restored GoEmotions comments, unchanged subreddit and thread keys.
- Predicate pairs: positive/negative and approval/disapproval; both key types (four queries).
- Oracle: local Qwen2.5-7B-Instruct Q4_K_M, greedy decoding, a true/false grammar,
  seed 17, no prompt caching, and at most four generated tokens.
- Before policy runs, materialize a complete fixed oracle reference for each
  predicate. This reference is available only to the validation harness.
- Every paid policy acquisition makes a new inference request. Returned model
  labels directly populate that policy's private facts and drive replanning.
  Check every returned label against the fixed reference. Any disagreement is
  retained as a reproducibility failure; never substitute a reference label.
- Policies: batched exact right-first, adaptive continuous per-side, and
  query-local cached joint greedy. Maximum acquisition batch 128. Audit 500/side.
- Two repetitions, with audit seeds 101 and 202 and rotated policy order.
  These are two independent audits and timing passes, not fixed-trace repeats.
- Report all four tasks, total requests, tokens, actual audit/execution batch
  latency, local overhead, SQL materialization, coverage, conflicts, and bag loss.
- Exclude model loading, warmup, complete reference acquisition, reference SQL
  construction, and independent validation from query cost. Charge incidence
  preprocessing, audit inversion, planning, bookkeeping, acquired facts, and
  returned-bag materialization. Publish excluded verification times separately.
- A fresh dense reference and per-acquisition live labels are distinct from the
  original immutable Llama-3.3-70B results. Do not pool oracle quality results.

## Planner scaling

- N = 1,024; 4,096; 16,384; 65,536; 131,072 unknown identities.
- K = 8 and 128, balanced sides, uniform and Zipf(1.5) key sizes.
- Fixed count-state construction, same total-positive intervals for both sides,
  no forced discards, allowance 1/10. Record the exact generated states.
- Three cold builds per cell, recording exact-profile time, heap time, process
  CPU, Python-traced peak allocation and maximum resident-set measurements.
  Warm exact-state cache lookup is measured separately and never called a cold
  planner measurement. Keep a 120-second per-build deadline with censored cases.
- Run DP only up to N=4,096; retain the same states for comparison. Scaling
  describes the planner kernel, not an end-to-end database/GPU benchmark.

## Certificate mechanism and independent audits

- Controlled certificate grid: N=128/512/2,048, K=2/8/32, uniform/Zipf(1.5)
  sizes, interval relative widths 0/.1/.3/.6/1, two seeds. Fix the total count
  midpoint at one half of each side, independently of width. Compare exact
  local profiles and the same deterministic greedy allocator under joint
  support and separate loss/total extrema. Publish every cell.
- Independent calibration: the same full 12 natural query roster as the paper,
  50 new audit seeds and budgets 125/500. Count one event per query/seed/budget,
  not per method or region. Report simultaneous count coverage and interval
  inconsistency after revealing all fixed labels. This is audit calibration,
  not a new complete-query loss study.

All drivers, inputs, raw records, source hashes and aggregation commands are
included. Historical source hashes remain historical provenance, while the
top-level artifact manifest binds the files distributed in this release.
