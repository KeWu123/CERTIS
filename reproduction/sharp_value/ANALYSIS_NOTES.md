The primary 64 executions and their archive equivalence checks passed unchanged.
The first post-processing pass tried to order a null same-evidence discard gap.
A null gap means the separate certificate has no certified superset of the
already irrevocable joint discards at that snapshot. The wrapper deliberately
retained that distinct state. Post-processing now counts these states separately
and orders only comparable greedy answers. No primary raw record was modified.
