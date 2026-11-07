# Native sharp-certificate evidence

The `results/` files preserve all 64 recorded seed-101 executions: 16 related
native queries and two audit budgets, comparing joint/separate with the same
allocator. They use the archived 70B fixed labels and report paid request
counts, not model inference latency. `pilot/` preserves the earlier diagnostic
and its recorded aggregation correction. Frozen sources are byte-exact;
portable entry points have distinct hashes in `portable_sources.json`.

From the repository root, without a model or GPU:

```
python reproduction/sharp_value/verify_results.py
python reproduction/sharp_value/run_native_sharp.py --pilot --output outputs/sharp_pilot
# Optional exact-arithmetic LP test; no replay or model inference.
python reproduction/sharp_value/test_fractional_relaxation.py
```

The second command makes fresh replay executions with a new source binding.
For the complete replay remove `--pilot`. Run analysis entry points against
that fresh `--output` directory using their `--results` option. Never resume a
historical output with portable source hashes. Fractional bounds concern
initial fixed-cache incident discards, not adaptive request optimality; orphan
counts require separate addition. Keep the post hoc diagnostic label.
