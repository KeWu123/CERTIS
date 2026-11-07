# Native independent-value study, frozen before execution

Date: 2026-11-07. Starting release: CERTIS artifact v3, commit
264d73022cf087431d133e32a34f0d2229aa815f.

## Question and complete primary matrix

Does replacing global separate extrema by the joint sharp certificate change
certified masks and complete paid endpoint requests on unchanged native queries?
Run every one of the existing 16 DBpedia native queries (the eight released
predicate pairs, in their declared order, each with l1 and l2 category keys),
audit budgets 125 and 500 per side, seed 101: 32 matched cells, two policies each.
No query, budget, or cell will be selected using outcomes. Report every cell,
including ties, negative request savings, and greedy reversals. This is a fresh
finite-tape execution, not live model inference or an end-to-end latency study.

## Controlled policy substitution

Both policies use the released target/tape, audit procedure and confidence
budget, endpoint cache semantics, degree acquisition order, logical batch size
128, cached-positive rescue, irreversible discards, feedback, greedy heap and
tie-breaking. Both use the immutable v3 guarded exact fast profile builder with
query-local memoization. No original source or historical record is overwritten.

For allowance p/q, joint support is max_z [q loss(z)-p total(z)]. Separate
support is q max_z loss(z)-p min_z total(z), summed globally over keys. A
separate local profile therefore uses the same exact builder with allowance 0,
then subtracts p times the minimum feasible total, a mask-independent constant.
The separate executor independently rechecks its selected mask with the original
joint kernel; its allocation decisions use only separate profiles. This follows
the existing constructed separate baseline, while retaining both support values.

## Snapshot diagnostics and interpretation

At every actual plan call, retain the complete side/key count states, paid count,
key sizes, unknown-side imbalance, interval widths, and known positives. On that
same evidence and forced-discard state, build both profile families and run the
same greedy heap. Evaluate each selected witness under both certificates, rather
than substituting the other family's potentially different minimum witness.
Record whether the joint-chosen mask satisfies H_joint <= 0 < H_separate, and
the difference in greedy discard cardinality at identical evidence. These are
distinct from the complete-policy request difference, since feedback trajectories
can diverge. Pointwise certificate dominance does not imply greedy dominance.

Checks: same audit records and ordered audit identities per matched cell;
pointwise joint <= separate for each actual chosen mask; returned SQL bag subset;
all actual labels inside intervals when coverage holds; bag loss <= 0.1 on that
event; immutable source hashes; fresh joint output equivalence to the published
seed-101 joint records (ordered paid ledger, facts, masks, batches and SQL counts).
If a check fails, preserve the failed record and diagnose before aggregating it.

## Reporting

Primary: pooled request totals and savings, query-level macro mean/median/worst
savings, and cell-level wins/ties/losses, split by key level and budget. Also
report initial discard gaps and the fraction of initial and all snapshots where
the selected joint mask needs coupling. Do not treat seeds or snapshots as
independent workloads. Illustrative snapshots, if useful, are deterministic:
largest positive same-mask support gap, largest negative greedy discard gap,
and largest request gain, with the selection rule disclosed and all rows saved.
CPU timing is diagnostic only because dual-certificate instrumentation adds work.
