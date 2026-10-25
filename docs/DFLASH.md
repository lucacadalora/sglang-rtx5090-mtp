# DFlash2 drafter on one RTX 5090 (October 2026)

[DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) is a 5-layer block-diffusion drafter trained for
Qwen3.8-27B: it proposes a whole block of 8 tokens in one pass (sliding-window attention over the last 2,048 target
tokens, a top-16 selector per position), and the target verifies the block. Its authors report, on one H200, 236
tok/s at one stream on GSM8K against 178.5 for Qwen's built-in 7-token MTP, and accepted lengths of 4.1-5.5 against
3.7-5.0. SGLang v0.5.20, the version we pin, already supports it (`--speculative-algorithm DFLASH`), so it can be
tried with no rebase. This page is what happened when we did, and why the result depends so much on the box.

## What we measured (home card, WSL2, 26 October 2026)

Same harness (`bench/ab_bench.py`), same machine, back to back. Prose replies (explanatory essays) at temperature 0,
the workload where drafters accept least. Raw rows: [bench/results/ab_results_2026-10-26.jsonl](../bench/results).

| | MTP head (production) | DFlash2, NVFP4 drafter |
|---|---|---|
| decode, 1 stream (server median / client) | 130.7 / 133.2 tok/s | **145.4 / 156.7** tok/s |
| accepted tokens per verify step, 1 stream | ~2.4 | 3.12 |
| most requests at once | 7 (783.8 tok/s) | 3 (334.6 tok/s) |
| KV pool | 176,031 tokens | 58,117 tokens |
| prefill, 8K / 32K prompt | 9,617 / 8,309 tok/s | 10,707 / 8,370 tok/s |
| prompt log-prob fingerprint vs reference | identical | identical |
| greedy 160-token reply vs reference | identical (675/675 chars) | departs at char 108 |

So: +11% on one stream by the server's median (+18% by the client), a quarter of the KV pool and less than half the
concurrency. The greedy departure is the same kind as MTP vs the plain model (verification batches positions
differently, which flips near-tie tokens); it needs the quality gates, not byte equality, but it means the drafter is
not a drop-in swap. Acceptance was well below the H200 figures: our target is NVFP4 (the drafter was trained against
bf16 hidden states), the drafter was a community NVFP4 conversion, and the workload was prose.

## Why the pool collapses: the verify scratch

On this hybrid model (48 Gated DeltaNet layers, 16 attention layers) a rejected draft token must not leave its
update in the recurrent state. The MTP profile solves that with ReplaySSM (`--enable-linear-replayssm-spec`), which
replays from small per-token records. In v0.5.20, ReplaySSM spec-verify with DFLASH is accepted only for KDA (Kimi
linear) models, so on Qwen3.8 the verify step keeps a full bf16 GDN state for every draft token of every running
request:

```
intermediate_ssm_state_cache = 48 layers x 1.5 MiB x block (8) x (max_running + 1)
                             = 2.25 GiB at 3 running, 4.5 GiB at 7     (boot log: "2.25GB" at 3)
```

On top of that, the drafter has its own KV pool with **one slot per target token** (the 2,048 window only narrows
what the drafter reads, not what is allocated): 5 layers x 8 KV heads x 256 x 1 byte = 10 KiB per token in fp8, next
to the target's 32 KiB. Draft weights are 1.60 GiB (NVFP4) or 3.6 GiB (bf16). With the ~23 GiB static budget a
Windows desktop leaves us, that is 58K tokens at 3 running, and 7 running does not fit at all.

`deploy/vastai/serve.sh` (profile `27b-dflash`) puts the scratch into its pool estimate; on the home card the
estimate came within ~2% of the allocated pool.

## What a headless card changes

A dedicated Linux box gives the model roughly 6 GB more (no desktop, no WDDM reserve, no WSL2 hidden memory).
`serve.sh`'s estimate for a headless card (32,111 MiB visible, default sizing, 22 state slots):

| arm | running | est. KV pool | window |
|---|---|---|---|
| MTP head | 7 | ~299K | 147,456 |
| DFlash2 NVFP4 | 3 | ~160K | 147,456 |
| DFlash2 NVFP4 | 7 | ~103K | ~100K |
| DFlash2 bf16 | 1 | ~139K | ~136K |
| DFlash2 bf16 | 3 | ~110K | ~107K |

The bf16 drafter also keeps the fused draft-KV path that the NVFP4 conversion disables. The default sizing is
conservative for DFLASH (the home boot still had 4.4 GB free after its CUDA graphs), so `MEM_FRACTION` can be raised
for more pool. All of this is estimated; [deploy/ubuntu](../deploy/ubuntu) has the matrix that measures it
(`ab_matrix.sh`), on prose, code and JSON workloads.

## Published numbers to compare against (RTX 5090, Qwen3.8-27B)

| source | setup | 1 stream | more streams |
|---|---|---|---|
| [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) | NVFP4 + DFlash2 bf16 drafter, 1 request, v0.5.19 | 4.92 ms/token (~203 tok/s), accept 4.29 | not given |
| [NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | NVFP4 + MTP (3 tokens) | 133 story, 205 code, 232 JSON | 440 at 4, 491 at 8 (corpus) |
| ours, home card | NVFP4 + MTP, k1 + k2 | 131-163 (prose) | 784-805 at 7 |

Different prompts, temperatures and counting methods; read them as orders of magnitude, not a ranking.

## What would remove the trade

1. **ReplaySSM spec-verify for Gated DeltaNet on the DFLASH commit path.** The scratch then disappears and DFlash2
   could run 7 at once with the full window. This is the real fix; it is a runtime patch in the style of k1/k2.
2. **Block size 4** (`DFLASH_BLOCK=4`) halves the scratch at the cost of the longest accepts.
3. **A newer SGLang** (v0.5.22 adds block verification and more speculative work) costs a rebase of every patch
   here: 206 upstream commits touch the nine files they anchor into.
