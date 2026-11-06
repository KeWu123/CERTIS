# Paper-to-artifact map

Run `python reproduce.py verify` to check every distributed file digest.
Run `python reproduce.py verify-paper` to independently recompute paid ledger,
token/phase and full/returned bag-count checks for the complete historical
matrices and to check the seven matched comparison totals.

Archived timings belong to their recorded environments. New executions create
new source/input bindings and timings. Never resume a different source version
into an existing output directory.

| Paper evidence | Distributed source records | Fresh execution |
|---|---|---|
| Certificate, Boolean lift, refinement, fixed-cache allocation | `code/` and mathematical checks | `python reproduce.py check` |
| Tables I/II: 1,376 primary executions | `reproduction/archive/primary/raw/` | `python reproduce.py run-primary --output outputs/primary` |
| Additional exact controls | `reproduction/archive/primary/extra_exact_controls/raw/` | `python reproduce.py run-extra --output outputs/extra` |
| Original all-mask DP, 112 executions including point controls | `reproduction/archive/primary/count_allocation/raw/` | `python reproduce.py run-count --output outputs/count` |
| Table III: four per-side policies, separate extrema, joint greedy, joint DP; 16 targets x 3 seeds | `reproduction/study/constructed_v1/`, `fixed_continuous_v2/`, `reproduction/completion/results/separate/`, greedy `full/raw/`, archived DP | `python reproduce.py run-matched --output outputs/matched` |
| Table IV: constructed and original native exact references | Primary count/extra records and `reproduction/archive/native/raw/` | `run-count`, `run-extra`, `run-native-original` |
| Fig. 3: matched allocation and 288 mechanism controls | Table III sources and `reproduction/archive/primary/mechanism_controls/raw/` | `python reproduce.py run-mechanisms --output outputs/mechanisms` |
| Fig. 4 and Table V: all 12 natural contrast queries and cost thresholds | `reproduction/study/native_v1/`, `native_cached_v1/`, token/batch ledgers | `python reproduce.py run --kind native --output outputs/native`; `run-cached` |
| Table V: all 16 native DBpedia category-key queries, 224 executions | `reproduction/completion/results/native_dbpedia/` | `python reproduce.py run-native-dbpedia --output outputs/native_dbpedia`; `analyze-native-dbpedia` |
| Concurrent fixed-trace local timing | `reproduction/study/local_live_v1/` | `python reproduction/study/local_batch_benchmark.py --server PATH --model PATH --output ../../outputs/timing` |
| Full 594-cell structural experiment and 3,168 geometry cases | `reproduction/archive/structural/` | `python reproduce.py run-structural --output outputs/structural`; `run-geometry` |
| Fig. 5(a): 20 planner configurations through N=131,072 | `reproduction/completion/results/counts/` | `python reproduce.py run-scaling --output outputs/scaling` |
| Fig. 5(b): 90 certificate configurations, two key permutations | Same count-results directory | `python reproduce.py run-certificate-grid --output outputs/certificate` |
| Calibration: 12 queries x 2 budgets x 50 seeds | Same count-results directory, `calibration.json` | `python reproduce.py run-calibration --output outputs/calibration` |
| Table VI: live closed-loop cost, four natural queries, two audits per policy | `reproduction/completion/results/closed_loop/` | `python reproduce.py run-closed-loop --server PATH --model PATH --output outputs/closed_loop`; `analyze-completion` |

`--pilot` selects one target/seed for fresh matrix runners. A primary pilot
executes 13 policies, a matched pilot all seven comparison methods, and a
structural pilot all six policies with explicit SQL bag checks. Full commands
regenerate results rather than rereading frozen plot inputs. Figures 1 and 2
describe the method and DP algorithm; their evidence is mathematical code and
checks rather than an experimental raw-result directory.

The final fixed-continuous policy spends the complete product-recall budget.
The historical attained-recall implementation remains in the earlier records;
the paper and fresh driver use `fixed_continuous_v2` semantics.

## Observation units and cost scopes

- The 48 constructed cells are 16 targets and three audit seeds, not 48
  independent query workloads.
- Calibration has 50 new seeds per natural query/budget. Methods do not count
  as independent audits. Query and budget groups can share data and draws;
  pooled event counts are descriptive.
- Controlled geometry uses constructed count intervals, not statistical
  audits. Exact certificate dominance is separate from greedy allocation.
- Cold memoized planning starts with an empty query cache. Equal states can
  hit within that query. Uncached construction and warm lookups are separate.
  Memory tracing is excluded from elapsed-time medians.
- Closed-loop inference uses its own Qwen labels and a fixed hash subset of
  2,048 comments. It does not change the archived Llama-3.3-70B oracle. One-slot
  GPU inference is serial within logical batches, distinct from the earlier
  128-slot concurrent fixed-trace scenario.
- DBpedia's released category fields are known keys correlated with the
  semantic predicates. They never become oracle facts. All eight pairs and
  both category levels are included; pooled gains do not imply gains on every
  query or against every exact control. `analysis.json` retains both batched
  directions and the retrospective smaller-direction diagnostic separately.
- The initial concurrent fixed-oracle attempt stopped on a label mismatch.
  `reproduction/completion/results/failed_parallel/` preserves that observation
  and manifest. `RUNTIME_ADDENDUM.md` declares the runtime correction.

Historical source identifiers describe original executions. Exported traces
retain `source_record_sha256`; distributed manifests bind portable metadata.
Source versions used by new studies are retained, including the version before
a Windows-only peak-memory measurement fix.
