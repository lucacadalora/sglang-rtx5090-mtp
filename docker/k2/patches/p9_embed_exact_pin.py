#!/usr/bin/env python3
"""k2 p9: pin the host-resident embedding table at its exact size (env SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN=1, default off).

torch's pin_memory() goes through the caching host allocator, which rounds each block up to a power of two:
measured 2026-09-24, the 248,320 x 5,120 bf16 table (2.368 GiB) pins 4,096 MiB, wasting 1.6 GiB of RAM (and of the
WDDM shared-memory budget). With the env var set, both pin sites of the embedding host offload (the k0 backport of
PR #37826) use exact_pinned_empty() from p8 instead. Same dtype, shape, values and host VA == device alias, so the
UVA gather kernel reads it unchanged. Requires p8 (exact_pinned.py).
"""
import pathlib

F = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/vocab_parallel_embedding.py")


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p9 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


assert pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/exact_pinned.py"
).exists(), "p9 needs p8 first"
t = F.read_text()
assert "SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN" not in t, "p9 already applied"
assert t.count("\nimport os\n") == 1, "p9: vocab_parallel_embedding.py no longer imports os at module level"
t = replace_once(
    t,
    "        self.weight.data = weight.data.pin_memory()\n",
    "        if os.environ.get(\"SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
    "            # k2 p9: exact-size pinned table (torch pin_memory rounds 2.37 GiB up to 4 GiB).\n"
    "            from sglang.srt.mem_cache.pool_host.exact_pinned import exact_pinned_empty\n"
    "\n"
    "            host = exact_pinned_empty(weight.shape, weight.dtype)\n"
    "            host.copy_(weight.data)\n"
    "            self.weight.data = host\n"
    "        else:\n"
    "            self.weight.data = weight.data.pin_memory()\n",
    "_pin_host_weight",
)
t = replace_once(
    t,
    "        host = torch.empty(weight.shape, dtype=weight.dtype, device=\"cpu\", pin_memory=True)\n",
    "        if os.environ.get(\"SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
    "            # k2 p9: exact-size pinned table (see _pin_host_weight).\n"
    "            from sglang.srt.mem_cache.pool_host.exact_pinned import exact_pinned_empty\n"
    "\n"
    "            host = exact_pinned_empty(weight.shape, weight.dtype)\n"
    "        else:\n"
    "            host = torch.empty(weight.shape, dtype=weight.dtype, device=\"cpu\", pin_memory=True)\n",
    "offload_weight_to_host",
)
F.write_text(t)
print("p9_embed_exact_pin applied")
