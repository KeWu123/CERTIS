# Native DBpedia paid-label closed loop

This protocol is fixed before collecting new Qwen oracle labels. It preserves
the released DBpedia rows, texts, and native l1/l2 equality keys. The first two
pairs in the released `common.DBPEDIA_PAIRS` declaration are agent/person and
person/athlete. Both pairs and both key levels are included, independent of
their observed labels or cost. The predicate statements are read from the
released dbpedia.yaml, including “describes a person or an organization” for
agent. Category metadata are keys, never oracle truth.

The default target contains all 5,000 rows on each side. An explicitly selected
smaller target ranks row identifiers by SHA256("certis-native-live-v1:" + row_id),
selects the first declared number, and restores original row order. Such a
target is a distinct workload with smaller key blocks. Choose the row count
before its dense reference and query execution. Preflight may estimate runtime
from a fixed 128-prompt roster, but cannot select predicates, keys, or rows by
observed classification or savings. All preflight records are retained.

The primary experiment was locked to the complete 5,000-row target, audit 500
per side, all four queries/methods, and both seeds before the new dense oracle
reference. Its full runtime was accepted after a throughput-only preflight;
no subset is used for this primary matrix.

Compare exact right-first, adaptive continuous per-side, global separate
extrema + greedy, and joint + greedy. Each run starts with a private empty
label cache and uncached SQL state. Use the same deterministic rank, greedy
tie-breaking, query-local memoization, and exact piecewise profile backend for
the separate and joint methods. The separate profile is
q*max_completion(T-B)-p*min_completion(T), with a common count-specific mask;
it is an adapted control, not an implementation of FDJ or BARGAIN. Fix recall
0.9, confidence 0.95, per-side audit budget 500, seeds 101/202, and two passes.
An explicit different audit budget belongs to a distinct bound manifest.
Rotate method order by query and pass. Keep both coarse and fine outcomes.

Use Qwen2.5-7B-Instruct Q4_K_M with a freshly measured, complete Boolean
reference for its three unique predicates. The original 70B arrays are never
used as Qwen labels. Actual paid outputs drive every audit and policy update.
Reference labels are accessible only to the validation harness. Every paid
answer is checked against the new reference; a mismatch stops the run and is
retained as failed evidence, without substituting a reference answer.

Pin model shards, executable and runtime DLL hashes, exact chat prompt hashes,
source hashes, full server command, decoding, grammar, parsing, target version,
and row roster. Use one server slot, no continuous batching, cache_prompt=false,
no prompt cache, temperature 0, seed 17, grammar true/false, n_predict=4, four
CPU threads, flash attention on, batch-size 2048, ubatch-size 512. Logical
acquisition groups contain at most 128 identities; inference inside a group is
serial. This is not a GPU batch speedup claim. Tokenize all canonical prompts
before any new labels; fail if any input plus output allowance exceeds the
fixed context. Never truncate an article. Check 128 fixed mixed prompts in
forward order and reverse order at logical caps 1/8/128 before the dense
reference. Report observed invariance, not universal determinism.

Record three distinct time scopes: (1) actual paid inference, by phase and
logical batch, including HTTP overhead; (2) execution cost = algorithm wall
time plus SQL incidence preprocessing, paid-fact ingestion, and actual
returned-bag materialization; (3) outer harness wall time. Report planner,
audit, execution bookkeeping, inline reference checks, independent full-label
loading/full-bag construction/EXCEPT ALL verification, corpus loading, model
startup, warmup, repeatability, and dense reference separately. The inline
reference-check monitor is explicitly measured and subtracted from the
execution scope; its unadjusted scope is also saved. Algorithm setup/cleanup
remain charged. No helper is acquired by these policies; assert this instead
of assigning unmeasured helper costs. Dense reference and independent SQL
validation are evaluation costs, not available to the deployed policy.
For every new run, materialize the returned bag before constructing the full
reference bag, using cost_sql.py. This avoids warming the returned query with
an excluded reference join. It is a separate measurement protocol from the
historical GoEmotions matrix; historical timings are not rewritten.

Save paid identities (including negatives), phases, labels, prompt hashes,
input/output tokens, raw response fields, batch wall times, acquisition order,
facts, masks, events, interval evidence, SQL bag counts, and frozen source
versions. Coverage is checked against the new finite reference; report
coverage failures, conflicts, empty answers, and losses. Two passes support
descriptive means/ranges, not significance claims. A replay self-test only
checks wiring and is explicitly not a live-cost observation. Preserve prior
70B replay results and prior GoEmotions Qwen experiments unchanged.
