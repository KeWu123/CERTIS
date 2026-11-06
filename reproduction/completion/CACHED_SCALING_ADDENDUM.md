# Cold memoization measurement

The initial scaling driver builds every key profile independently and separately
measures warm exact-state lookups. The practical executor also reuses equal
states across different keys within the first query. Thus even a cold query
can have profile-cache hits, especially with equal-size uniform blocks.

Measure all 20 scaling cells again with three timed cold query-local caches
and one separate allocation-tracing pass. Each cache starts empty, has 65,536
entries, and wraps the unmodified exact profile kernel. Include the heap in
elapsed time. Publish misses/hits, distinct states, support and discard count;
compare those outcomes with the uncached kernel measurements. Keep both sets
of observations, with distinct scopes. No speed ratio is inferred from the two
sets of wall times because their surrounding processes can differ.
