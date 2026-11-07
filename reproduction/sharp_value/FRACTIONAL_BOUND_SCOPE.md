# Bound input and accounting convention

`fractional_upper_bound` expects monotone local `CountProfile.values` over
contiguous extra-discard counts, with `forced` counting incident identities
already discarded. Its return includes those forced incident identities.
Monotonicity is checked in the function. For a general target, orphan unknowns
must be added separately to both feasible counts and the upper bound. Every row
of the current native initial-state diagnostic has zero orphans and zero forced
discards, explicitly checked against the original executor mask counts.

The analysis result is bound to the unchanged executed source and original
supplemental protocol in `results/fractional_provenance.json`. This sidecar was
added after the calculation; it adds provenance without changing numerical
results. The independent cost_audit agent also checked 160 exact LP vertex
enumerations, including signed bases and forced counts, and found equality
with the fractional function. That check is supplementary to the recorded
2,000-case integer-enumeration upper-bound validation.
