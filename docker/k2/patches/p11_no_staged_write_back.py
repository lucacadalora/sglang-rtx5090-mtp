#!/usr/bin/env python3
"""k2 p11: one-kernel page_first KV backup (env SGLANG_HICACHE_NO_STAGED_WRITE_BACK=1, default off).

On the hybrid GDN model HiCache must use the page_first layout (v0.5.20 MambaPoolHost asserts page_first /
page_first_direct). The page_first KV backup then goes through the staged write-back JIT kernel, which relayouts
64 pages at a time into a device staging buffer and copies every page to host with cudaMemcpyBatchAsync only when a
page is >= 128 KiB. At page_size 1 a page is 17 KiB (16 target + 1 MTP layer x 1 KiB), so it falls back to one
cudaMemcpyAsync per token for K and another for V (~120K calls for a 60K-token prompt). With the env var set, the
MHA host pool reports can_use_write_back_jit = False and keeps the single-kernel transfer_kv_all_layer_lf_pf backup;
start_writing() then moves host indices to the GPU for it (the flag's only other effect; both paths exist upstream).
"""
import pathlib

F = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/mha.py")


def replace_n(text, old, new, what, n_expected):
    n = text.count(old)
    assert n == n_expected, f"p11 {what}: expected {n_expected} anchors, found {n}"
    return text.replace(old, new)


# Two MHA host-pool classes in v0.5.20 share this staging init verbatim (the plain pool and the split-head-dim
# pool); both get the same gate so the flag means the same thing whichever class a model builds.
t = F.read_text()
assert "SGLANG_HICACHE_NO_STAGED_WRITE_BACK" not in t, "p11 already applied"
t = replace_n(
    t,
    "    def _init_write_back_staging_buffers(self):\n"
    "        self.staging_page_capacity = 0\n"
    "        self.staging_token_capacity = 0\n"
    "        self.staging_k_buffer = None\n"
    "        self.staging_v_buffer = None\n"
    "        self.can_use_write_back_jit = False\n"
    "        if self.layout != \"page_first\" or (_is_npu or _is_xpu or _is_mps):\n"
    "            return\n",
    "    def _init_write_back_staging_buffers(self):\n"
    "        self.staging_page_capacity = 0\n"
    "        self.staging_token_capacity = 0\n"
    "        self.staging_k_buffer = None\n"
    "        self.staging_v_buffer = None\n"
    "        self.can_use_write_back_jit = False\n"
    "        if self.layout != \"page_first\" or (_is_npu or _is_xpu or _is_mps):\n"
    "            return\n"
    "        # k2 p11 (env SGLANG_HICACHE_NO_STAGED_WRITE_BACK=1, default off): keep the one-kernel page_first\n"
    "        # backup. At page_size 1 a page is under the staged path's 128 KiB batch threshold, so it would issue\n"
    "        # one cudaMemcpyAsync per token for K and for V.\n"
    "        import os as _os\n"
    "\n"
    "        if _os.environ.get(\"SGLANG_HICACHE_NO_STAGED_WRITE_BACK\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
    "            return\n",
    "MHA _init_write_back_staging_buffers",
    2,
)
F.write_text(t)
print("p11_no_staged_write_back applied")
