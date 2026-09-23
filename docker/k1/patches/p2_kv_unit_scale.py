#!/usr/bin/env python3
"""k1 p2 (plan rank 2): skip the FP8 KV-write division when the layer's k/v scales are exactly 1.0.

Why: the checkpoint ships no k_scale/v_scale tensors, so kv_cache.py:59-83 sets both to 1.0 (tensor and
k_scale_float). MHATokenToKVPool.set_kv_buffer (memory_pool.py ~2563-2566) still runs cache_k.div_(1.0) and
cache_v.div_(1.0) before the e4m3 cast: 2 DivFunctor kernels (and 2 graph nodes) per attention layer, 32 per cycle.

Bit-exactness: div_ computes bf16(float(x) / 1.0f) (or x * 1.0f). For every non-NaN x (finite, +-inf, signed
zeros, denormals) x / 1.0 == x exactly in IEEE-754: the quotient is representable, so round-to-nearest returns x;
denormals are preserved without FTZ and, even with FTZ, the following e4m3 cast rounds them to the same signed zero;
the bf16 round-trip of an exact bf16 value is the identity. Checked exhaustively: all 65,282 non-NaN bf16 bit
patterns give identical bf16 bits and identical e4m3 bytes with and without the division (CPU and GPU, see
tests/). The only difference is for NaN inputs: the division canonicalises the NaN, so the e4m3 NaN byte may flip
between 0x7F and 0xFF (both NaN); a NaN in K/V already poisons attention either way. So the stored FP8 KV is
bit-identical for every real input, i.e. the change is lossless.

Env (read on first use, then cached per backend / pool object; unset = today's behaviour, same kernels):
  SGLANG_KV_SKIP_UNIT_SCALE_DIV=1   enable (default off).

Edits:
  1. flashinfer_backend.py FlashInferAttnBackend._kv_write_scales (~933-936): return (None, None) when enabled,
     the KV quant method is "unquantized" (plain dtype-cast FP8/bf16 path, where None means "no division"; the
     quantized FP4 path gives None a different meaning, so it is excluded), both scale tensors exist and
     layer.k_scale_float == layer.v_scale_float == 1.0. Draft (bf16 MTP) layers have no scales and are unchanged.
  2. memory_pool.py MHATokenToKVPool.set_kv_buffer (~2563-2566), belt and braces for other attention backends that
     pass layer.k_scale directly: when enabled, a scale that IS the layer's own k_scale/v_scale tensor while
     layer.k_scale_float == layer.v_scale_float == 1.0 is treated as None. Other callers/scales are untouched.
Falls back to the division automatically if a future checkpoint ships real (non-1.0) scales.

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server.
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
FIB = ROOT / "srt/layers/attention/flashinfer_backend.py"
POOL = ROOT / "srt/mem_cache/memory_pool.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p2 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


# ---- 1. flashinfer backend ----
t = FIB.read_text()
assert "_k1_kv_skip_unit_scale_div" not in t, "p2 already applied (flashinfer_backend)"
assert t.count("\nimport os\n") == 1, "p2: flashinfer_backend.py no longer imports os at module level"
t = replace_once(
    t,
    "    def _kv_write_scales(self, layer: RadixAttention):\n"
    "        if self.kv_cache_quant_method.needs_global_scale():\n"
    "            return None, None\n"
    "        return layer.k_scale, layer.v_scale\n",
    "    def _kv_write_scales(self, layer: RadixAttention):\n"
    "        if self.kv_cache_quant_method.needs_global_scale():\n"
    "            return None, None\n"
    "        # k1 p2 (env SGLANG_KV_SKIP_UNIT_SCALE_DIV=1, default off): x / 1.0 is exact, so\n"
    "        # unit k/v scales need no div_ before the FP8 cast (2 kernels per layer saved).\n"
    "        skip_unit = getattr(self, \"_k1_kv_skip_unit_scale_div\", None)\n"
    "        if skip_unit is None:\n"
    "            skip_unit = os.environ.get(\n"
    "                \"SGLANG_KV_SKIP_UNIT_SCALE_DIV\", \"0\"\n"
    "            ).strip().lower() in (\"1\", \"true\", \"yes\") and (\n"
    "                getattr(self.kv_cache_quant_method, \"name\", None) == \"unquantized\"\n"
    "            )\n"
    "            self._k1_kv_skip_unit_scale_div = skip_unit\n"
    "        if (\n"
    "            skip_unit\n"
    "            and layer.k_scale is not None\n"
    "            and layer.v_scale is not None\n"
    "            and layer.k_scale_float == 1.0\n"
    "            and layer.v_scale_float == 1.0\n"
    "        ):\n"
    "            return None, None\n"
    "        return layer.k_scale, layer.v_scale\n",
    "_kv_write_scales",
)
FIB.write_text(t)

# ---- 2. MHA pool (belt and braces for backends that pass layer.k_scale directly) ----
p = POOL.read_text()
assert "_k1_kv_skip_unit_scale_div" not in p, "p2 already applied (memory_pool)"
assert p.count("\nimport os\n") == 1, "p2: memory_pool.py no longer imports os at module level"
p = replace_once(
    p,
    "        if cache_k.dtype != self.dtype:\n"
    "            if k_scale is not None:\n"
    "                cache_k.div_(k_scale)\n"
    "            if v_scale is not None:\n"
    "                cache_v.div_(v_scale)\n"
    "            cache_k = cache_k.to(self.dtype)\n"
    "            cache_v = cache_v.to(self.dtype)\n"
    "\n"
    "        if self.store_dtype != self.dtype:\n"
    "            cache_k = cache_k.view(self.store_dtype)\n",
    "        if cache_k.dtype != self.dtype:\n"
    "            # k1 p2 (env SGLANG_KV_SKIP_UNIT_SCALE_DIV=1, default off): the layer's own unit\n"
    "            # scales divide exactly, so drop them (same bytes, 2 fewer kernels per layer).\n"
    "            _k1_skip = getattr(self, \"_k1_kv_skip_unit_scale_div\", None)\n"
    "            if _k1_skip is None:\n"
    "                _k1_skip = os.environ.get(\n"
    "                    \"SGLANG_KV_SKIP_UNIT_SCALE_DIV\", \"0\"\n"
    "                ).strip().lower() in (\"1\", \"true\", \"yes\")\n"
    "                self._k1_kv_skip_unit_scale_div = _k1_skip\n"
    "            if (\n"
    "                _k1_skip\n"
    "                and layer is not None\n"
    "                and getattr(layer, \"k_scale_float\", None) == 1.0\n"
    "                and getattr(layer, \"v_scale_float\", None) == 1.0\n"
    "            ):\n"
    "                if k_scale is not None and k_scale is getattr(layer, \"k_scale\", None):\n"
    "                    k_scale = None\n"
    "                if v_scale is not None and v_scale is getattr(layer, \"v_scale\", None):\n"
    "                    v_scale = None\n"
    "            if k_scale is not None:\n"
    "                cache_k.div_(k_scale)\n"
    "            if v_scale is not None:\n"
    "                cache_v.div_(v_scale)\n"
    "            cache_k = cache_k.to(self.dtype)\n"
    "            cache_v = cache_v.to(self.dtype)\n"
    "\n"
    "        if self.store_dtype != self.dtype:\n"
    "            cache_k = cache_k.view(self.store_dtype)\n",
    "MHATokenToKVPool.set_kv_buffer div",
)
POOL.write_text(p)
print("p2_kv_unit_scale applied")
