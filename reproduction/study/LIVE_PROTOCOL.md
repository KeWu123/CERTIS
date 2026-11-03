# Local batch timing scenario

Use the existing Qwen2.5-7B-Instruct Q4_K_M GGUF shards, llama.cpp build 11491 (9b4ed0ca5), and RTX 4090 D. This is a local timing scenario, separate from the immutable Llama-3.3-70B correctness tape. Live outputs are recorded but never substitute for the labels that chose the archived acquisition schedules.

Choose the first declared natural task (positive/negative, subreddit), seed 101, budget 500 per side. Replay every actual batch of the exact right-first, adaptive continuous per-side and joint-greedy traces. The selection is fixed before their full-matrix outcomes are available. A batch is never moved across an adaptive feedback boundary. Audit calls are included. No remote API, model download or paid service is used.

Use 128 server slots, 65,536 total context, all model layers on GPU, CUDA flash attention, no prompt/KV reuse, greedy decoding and a true/false grammar with maximum four generated tokens. Reject any prompt truncation or failed/empty response. Warm up one batch of 128 before timing. A controlled throughput probe uses the same 128 prompts, client caps 1/8/32/128, and three rotated repetitions; report all measurements. Full trace replays use cap 128 and one complete pass per policy in a fixed interleaved batch schedule. They are measured trace times rather than repeated full-query confidence intervals.

Sum each trace's measured request wall time, plus its measured metadata/planning/audit/returned-SQL overhead. Exclude model loading, experimental reference-label loading, full-reference SQL and EXCEPT ALL verification. Report those exclusions. Because the live model differs, this does not establish a Llama-70B end-to-end speedup or a Qwen-specific recall guarantee.
