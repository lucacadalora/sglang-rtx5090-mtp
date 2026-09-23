# Raw A/B results

One JSON object per `bench/ab_bench.py` run, in the order they ran, with a `config` note added by hand. Launch
arguments and file paths were removed before publishing.

- `ab_results_2026-09-23.jsonl`: the MTP and host-embedding work (23 September 2026).
- `ab_results_2026-09-24_k1.jsonl`: the k1 kernel/runtime patches (24 September 2026, 03:20-04:40), with extra
  fields `flags` (k1 env switches on), `cycle_ms_bs1_median` and `accept_len_bs1` from `bench/accept_len.sh`.

| Field | Meaning |
|---|---|
| `decode1_server`, `decodeN_server` | Median of the scheduler's `gen throughput` (tok/s, all streams together) for batches with exactly N running requests; `_samples` = log lines used |
| `decode1_client` | Client-side decode rate for the same single-stream replies (reads low: the idle GPU ramps up) |
| `ttft_max_at_N` | Slowest time to first token among the N parallel requests, seconds |
| `power_w_mean`, `joules_per_out_token` | GPU board power sampled every 0.5 s during the decode tests; energy / output tokens |
| `prefill_8000_*`, `prefill_32000_*` | Uncached prompts of about 6K and 24K tokens: prompt tokens and tokens per second |
| `pool` | KV cache pool in tokens (`sglang:max_total_num_tokens`) |
| `fp_*`, `greedy_*` | Correctness against the most recently saved reference for that model: log-prob difference over the last 128 prompt positions, and common prefix of a 160-token greedy reply |
| `cycle_ms_bs1_median` | Median single-stream MTP cycle: accept_len x running / gen_throughput (lower is faster) |

The 23 September reference was re-saved several times during the evening (rows without `fp_*` fields saved one),
so compare `fp_*` only between neighbouring rows. `base-27b-live` and `base-27b-quiet` are the paging-degraded runs
that led to the WDDM finding; the fair plain baseline is `base-27b-clean` and the `opt1-27b` runs.

On 24 September the machine itself drifted: near-identical configurations measured a single-stream cycle between
15.8 and 21.1 ms within one hour (desktop apps and Windows background work share the GPU and CPU). Compare only
rows that ran next to each other: `k1-safe` vs `k1-off-2` (+15.5% single stream) and `k1-off-2` vs `k1-prod`
(+1.7%). The honest reading is "faster, by an amount this window could not pin down".

Temp-0 equality runs (`bench/tok_equal.py`, 50 prompts + 3 at 32K/64K/120K tokens):

| Comparison | Equal | Note |
|---|---|---|
| production vs k1-off, first run after launch | 43/53 | the ten long prompts differ: first-run prefill tactic tuning |
| production vs k1-off, second run | 53/53 | warm server is deterministic |
| production vs k1-quick (p1 p2 p3 p7) | 8/53 | p1 moves greedy text (see docs/K1_KERNELS.md) |
| k1-quick vs k1-all (+ p6 seam) | 53/53 | the seam is exact |
| production before k1 vs k1-prod (p2 p3 p7 p6) | 53/53 | what ships |
