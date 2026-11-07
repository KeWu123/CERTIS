# Native paid-label live evidence

`results/` preserves every actual paid acquisition, all three independent Qwen
reference predicates, repeatability checks, returned-first SQL counts and
source snapshots. `native_live_analysis.json` is the complete checked report.
`preflight_initial/` preserves the earlier preflight; `hardware_monitor.jsonl`
preserves its recorded GPU metrics. PID files and server diagnostic logs are
excluded; paid responses/tokens/timings, failures and scientific inputs remain.

The four related queries, four methods and two audit seeds produce 32 runs,
not 32 independent workloads. Single-slot inference is serial inside logical
batches. New actual cost is distinct from both 70B replay and older discussion
live measurements. Reference construction/validation, loading and warmup are
excluded from the complete query scope; every raw timing scope is retained.

From the repository root, recheck all supplied evidence without a model:

```
python reproduction/native_live/analyze_native_live.py --source reproduction/native_live/results --output outputs/native_live_checked.json
python reproduction/native_live/test_cost_sql.py
python reproduction/native_live/test_aggregation.py
python reproduction/native_live/plot_submission_evidence.py --require-live
python reproduction/native_live/build_evidence_table.py
# Optional secondary post hoc initial-cache diagnostic; no model/GPU needed.
python reproduction/native_live/diagnose_live_initial.py
```

Actual inference requires separately supplied llama.cpp/model files and a GPU:

```
python reproduction/native_live/native_live.py preflight --server PATH_TO_LLAMA_SERVER --model PATH_TO_GGUF --output outputs/native_preflight
python reproduction/native_live/native_live.py run --server PATH_TO_LLAMA_SERVER --model PATH_TO_GGUF --output outputs/native_new
```

The model is not included. Defaults preserve the full 5000-row, two-pair,
l1/l2, audit-500, two-seed matrix. Fresh runs bind portable sources and do not
resume the original binding. Executed source_versions are archived provenance,
not runnable entry points. Generic local runtime paths can occur inside those
unaltered historical bytes; runnable defaults use explicit model/server args.

## Same-Qwen initial-cache diagnostic

The unchanged `LIVE_INITIAL_RESULTS.md` and `live_initial_diagnostic.json` retain all eight post hoc states. All eight concrete joint masks require coupling, but none exceeds the separate fractional upper bound: the 0/8 result is inconclusive about optimal-cardinality equality. The old 70B 15/32 result remains in its distinct oracle/evidence domain.

`initial_checks_execution_public.json` is explicitly a presentation copy of the operational execution log. It preserves original elapsed seconds, exit codes, executed-source hashes and numerical stdout, while omitting private absolute interpreter/script paths. Its command schematics are portable examples, not a claim that those commands were executed. The original log remains private and is bound by SHA; scientific diagnostic/raw records are unmodified.
