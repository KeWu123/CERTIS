# Secondary post hoc diagnostic: initial live-Qwen certificates

`diagnose_live_initial.py` is prepared for execution only after all 32 actual
native policy runs and the independent checked analysis report pass. It has no
partial mode. It never calls the model, reruns a policy, changes production raw
records, or adds diagnostic work to measured policy cost. The script applies to
all four declared native queries and both seeds, eight joint/separate cells,
regardless of whether the result favors joint.

Reconstruction uses only the first 1,000 paid audit ledger facts (500 per side).
For each cell, `allUnknownAfterAudit - initial_retained_unknown` reconstructs the
joint initial mask. The script checks this against `initial_discarded`, the
initial certified-discard count, initial event count witnesses and support.
The sorted 5,000-row dataset roster and original l1/l2 category memberships
reconstruct the side/key regions. For each side, `SideCounts(lower, upper, h, n,
forced_discard=0)` uses the audit region bounds, audit positives h and unknown
count n. No final execution labels or original 70B labels enter that state.

The script recomputes both certificates on the same concrete joint mask,
independently checks joint support with the original polygon kernel, and checks
the actual separate initial mask on the same audit evidence. It then uses the
already verified `build_profile(..., separate=True)` and `fractional_upper_bound`
with exact `Fraction`. Native memberships contain every identity exactly once,
so orphan count is asserted zero. The bound counts incident discards including
forced counts; this initial study also asserts all forced counts are zero.

The schema deliberately separates four quantities:

- `actual_joint_initial`: concrete initial mask IDs, incident/total discard
  count, counts per key, same-mask joint/separate support and acceptance flag.
- `actual_separate_initial`: the actually executed separate policy's initial
  mask and discard counts, not its unknown cardinality optimum.
- `separate_fractional_relaxation`: exact rational upper bound, its integer
  floor and `max(0, G_joint - floor(U_sep))`, the proven certificate advantage
  independently of greedy. This bound is not an executable mask or an adaptive
  request-cost optimum.
- `complete_measured_costs`: the actual same-case/same-seed separate and joint
  request counts and complete seconds from the independently checked Qwen rows.
  Positive gaps favor joint. This descriptive linkage does not prove that an
  initial cardinality advantage caused the observed complete cost difference.

`key_count_states`, audit facts, mask IDs and raw hashes retain the entire
reconstruction. The result embeds production/diagnostic source SHA maps,
manifest/summary/checked-analysis hashes, all 32 checked raw hashes and the 16
raw hashes actually used for initial joint/separate comparison. The companion
`live_initial_diagnostic_provenance.json` also binds the result bytes.

Defaults:

```powershell
python research/icde2027_native_live_20261107/diagnose_live_initial.py
```

Input: `outputs/full5000_v1/`, checked `native_live_analysis.json`.
Output: `live_initial_diagnostic.json` and
`live_initial_diagnostic_provenance.json` in the native-live research directory.
Existing diagnostic files are accepted only if their bytes are identical.

This script was written during the live run and was not executed or imported.
Only syntax can be checked until the complete checked 32-run gate is satisfied.
