# Fixed-oracle runtime correction

The initial 128-slot closed-loop attempt was stopped when an acquired label
differed from the complete reference for the same prompt, model and decoding
settings. The mismatch and initial manifest are retained. No reference label
was substituted, and no query result from that attempt is used as evidence.

The new run uses one server slot and disables continuous batching. Logical
acquisition batches still contain at most 128 identities, but GPU inference is
serial within each logical batch. This is a distinct cost scenario from the
previous 128-slot fixed-trace measurements.

Before new oracle labels are collected, fix a 2,048-comment subset by ranking
the released row identifiers with SHA256("certis-live-v2:" + row_id), choosing
the first 2,048 and preserving their original order. Retain their original
subreddit/thread memberships. The four-query roster, methods, audit budgets,
and seeds stay the same. The reduced live target is declared explicitly; the
full 4,998-comment archived workload is unchanged.

Check label invariance under request ordering and logical batch sizes 1/8/128
before the new full reference. Then check every paid live answer against that
new reference during all 24 executions. This checks the observed workload;
it is not a universal determinism theorem for the inference implementation.
