# Supplemental diagnostic, declared during the primary run

The primary protocol and complete 32-cell matrix are unchanged. The first
completed rows show different greedy discard counts. Such a difference alone
does not establish a gap between the two certificate families' cardinality
optima, because neither heap is globally optimal.

As a secondary, post hoc diagnostic, compute a fractional upper bound on the
separate certificate's fixed-cache maximum discard count at **all 32 initial
snapshots**, without selecting snapshots using their gains. For each key use
options (extra discards, phi(g)-phi(0)), build the lower convex hull and merge its
edges by exact rational marginal cost. Spend the signed starting credit
-sum(phi(0)), allowing a fractional final edge. The floor of this rational value
plus forced discards is an upper bound on every integer feasible separate plan.

Reason: this relaxes each key's one-option constraint to a convex combination.
Every integer plan remains feasible. The increasing hull slopes permit the
fractional optimum by merged edge order. Negative starting supports are retained
as global credit. The bound is not an executable mask. If joint greedy's feasible
discard count exceeds this upper bound, the cardinality advantage is independent
of the greedy heuristic. Otherwise the diagnostic is inconclusive; no equality
with the separate optimum is asserted. The snapshot objective is not the full
adaptive request objective. Use integer arithmetic and Fraction throughout.

First validate the upper bound against brute-force integer optima on 2,000
deterministically generated small monotone profiles, including zero-cost and
nonconvex increments. Run the native secondary diagnostic after the primary CPU
matrix, to avoid slowing the main result or live cost measurements.
