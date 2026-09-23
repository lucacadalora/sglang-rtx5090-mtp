#!/usr/bin/env python3
"""k1 p7 (plan rank 7): enable the existing SiLU+mul+FP4-quant fusion in the 16 full-attention layers' MLPs.

Why: Qwen3_5LinearDecoderLayer (48 GDN layers) calls _maybe_enable_silu_fp4_quant_fusion(self.mlp) at
qwen3_5.py:976, so their dense MLPs run one FlashInfer silu_and_mul_scaled_nvfp4_experts_quantize kernel
(cvt_fp16_to_fp4) and feed down_proj a prequantized (fp4, scale) tuple. Qwen3_5AttentionDecoderLayer builds the
same Qwen2MoeMLP at :1192-1199 without that call, so its 16 MLPs still run act_and_mul + quantize_with_block_size
(2 kernels and 2 graph nodes per layer instead of 1).

Env (read once per attention layer at __init__ time; unset = today's code path, same kernels, same graphs):
  SGLANG_ATTN_MLP_SILU_FP4_FUSION=1   enable for the attention-layer MLPs (default off in image k1, so the
                                      flags-off relaunch is an exact no-op).
The existing global kill switch SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1 still wins (the helper checks it first).
The helper returns early unless gate_up_proj and down_proj both use ModelOptFp4LinearMethod in w4a4 mode, so the
bf16 MTP draft layer (quant_config=None) and any non-NVFP4 checkpoint are untouched.
Numerics: same kernel the GDN layers already run at M=4/20 in production; its bit-exactness note
(qwen2_moe.py:243-245) was only checked for M 64..8192, so FP4 rounding ties may differ from act_and_mul+fp4_quantize
(gate: ab_bench fp_mean_abs_diff <= 0.005, greedy 675/675 or a proven near-tie).

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server.
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
QWEN = ROOT / "srt/models/qwen3_5.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p7 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


t = QWEN.read_text()
assert "SGLANG_ATTN_MLP_SILU_FP4_FUSION" not in t, "p7 already applied"
# the helper this patch calls must still exist with its v0.5.20 name
assert t.count("def _maybe_enable_silu_fp4_quant_fusion(mlp: nn.Module) -> None:\n") == 1, "p7: helper missing"
t = replace_once(
    t,
    "                prefix=add_prefix(\"mlp\", prefix.replace(\".self_attn\", \"\")),\n"
    "            )\n"
    "            is_layer_sparse = False\n",
    "                prefix=add_prefix(\"mlp\", prefix.replace(\".self_attn\", \"\")),\n"
    "            )\n"
    "            # k1 p7 (env SGLANG_ATTN_MLP_SILU_FP4_FUSION=1, default off): same fused\n"
    "            # SiLU+mul+FP4-quant as the GDN-layer MLPs (mirrors the call at :976).\n"
    "            if os.environ.get(\"SGLANG_ATTN_MLP_SILU_FP4_FUSION\", \"0\").strip().lower() in (\n"
    "                \"1\",\n"
    "                \"true\",\n"
    "                \"yes\",\n"
    "            ):\n"
    "                _maybe_enable_silu_fp4_quant_fusion(self.mlp)\n"
    "            is_layer_sparse = False\n",
    "attention-layer dense MLP",
)
QWEN.write_text(t)
print("p7_silu_fp4_attn_mlp applied")
