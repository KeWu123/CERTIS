# Native DBpedia category-key workload

This exploratory extension is declared after the existing constructed and
GoEmotions studies. Existing outcomes are known. No new category-key policy
results have been inspected when declaring this roster.

- Use all 5,000 released DBpedia articles on both sides, with distinct
  predicate/row oracle identities. Same-article derivations are included.
- Use the original `l1` and `l2` category fields as two known equality-key
  relations. Do not manufacture membership or choose categories by outcomes.
  Category metadata are correlated with the semantic predicates; this is an
  explicit workload characteristic, not additional acquired oracle facts.
- Retain all eight existing DBpedia predicate pairs, both key levels, including
  zero-answer cases. This gives 16 natural queries and 10,000 identities each.
- Oracle: the unchanged released Llama-3.3-70B labels. Every algorithmic label
  acquisition uses the charged private Oracle cache. Category fields never
  become oracle labels or known-positive/negative facts.
- Methods: batched exact left/right; adaptive continuous per-side; joint greedy
  with empty per-query exact-state memoization caches. Logical batch size 128.
- Audits 125/500 per side, seeds 101/202/303, allowance 1/10, failure budget 1/20.
  There are 32 exact plus 192 approximate complete-query executions, 224 total.
- All queries materialize SQL bags and independently check subbag inclusion
  and loss against raw-parquet reference SQL. Retain coverage failures,
  inconsistencies, regressions and empty answers.
- Report complete audit/execution requests, input/output tokens, actual batch
  boundaries, planning/SQL costs, and pooled break-even latency per batch
  against batched exact right-first. These are immutable-tape experiments,
  not live 70B latency measurements.
