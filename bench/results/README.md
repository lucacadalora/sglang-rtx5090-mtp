# Raw A/B results, 23 September 2026

One JSON object per `bench/ab_bench.py` run, in the order they ran, with a `config` note added by hand. Launch
arguments and file paths were removed before publishing.

| Field | Meaning |
|---|---|
| `decode1_server`, `decodeN_server` | Median of the scheduler's `gen throughput` (tok/s, all streams together) for batches with exactly N running requests; `_samples` = log lines used |
| `decode1_client` | Client-side decode rate for the same single-stream replies (reads low: the idle GPU ramps up) |
| `ttft_max_at_N` | Slowest time to first token among the N parallel requests, seconds |
| `power_w_mean`, `joules_per_out_token` | GPU board power sampled every 0.5 s during the decode tests; energy / output tokens |
| `prefill_8000_*`, `prefill_32000_*` | Uncached prompts of about 6K and 24K tokens: prompt tokens and tokens per second |
| `pool` | KV cache pool in tokens (`sglang:max_total_num_tokens`) |
| `fp_*`, `greedy_*` | Correctness against the most recently saved reference for that model: log-prob difference over the last 128 prompt positions, and common prefix of a 160-token greedy reply |

The reference was re-saved several times during the evening (rows without `fp_*` fields saved one), so compare
`fp_*` only between neighbouring rows. `base-27b-live` and `base-27b-quiet` are the paging-degraded runs that led to
the WDDM finding; the fair plain baseline is `base-27b-clean` and the `opt1-27b` runs.
