#!/usr/bin/env python3
"""k2 p8: HiCache host pools from an exact cudaHostAlloc (env SGLANG_HICACHE_HOST_ALLOC=cuda, default off).

Measured on this box (kern/hicache/hicache_wsl_probe.py, 2026-09-24): CanUseHostPointerForRegisteredMem = 0 and the
installed sgl_kernel lacks get_device_accessible_ptr, so the stock pool (mmap + cudaHostRegister, host VA != device
alias) would hand raw host addresses to the GPU. cudaHostAlloc memory has host VA == device alias; the JIT all-layer
backup, JIT one-layer load, AOT all-layer/per-layer and AOT per-layer-mla (GDN state) paths were all byte-exact on it.

Edits (each anchor must match exactly once; a changed upstream file fails the build):
  1. installs exact_pinned.py as sglang/srt/mem_cache/pool_host/exact_pinned.py
  2. common.py: adds alloc_with_cuda_host_alloc() next to alloc_with_pin_memory() and, only when the env var is
     set, makes it the default entry of ALLOC_MEMORY_FUNCS (npu/musa entries untouched). The buffer carries an
     empty registered-ranges list so destroy() does not try to cudaHostUnregister it.
"""
import pathlib
import shutil

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
HOSTDIR = ROOT / "srt/mem_cache/pool_host"
COMMON = HOSTDIR / "common.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p8 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


dst = HOSTDIR / "exact_pinned.py"
assert not dst.exists(), "p8 already applied (exact_pinned.py exists)"
shutil.copyfile(pathlib.Path(__file__).with_name("exact_pinned.py"), dst)

t = COMMON.read_text()
assert "alloc_with_cuda_host_alloc" not in t, "p8 already applied (common.py)"
assert t.count("\nimport os\n") == 1, "p8: common.py no longer imports os at module level"
t = replace_once(
    t,
    "@lru_cache(maxsize=1)\ndef _resolve_device_accessible_ptr_fn():\n",
    "def alloc_with_cuda_host_alloc(\n"
    "    dims: tuple,\n"
    "    dtype: torch.dtype,\n"
    "    device: str,\n"
    "    pin_memory: bool,\n"
    "    allocator: HostTensorAllocator,\n"
    "    registration_granularity_bytes: int | None = None,\n"
    ") -> torch.Tensor:\n"
    "    \"\"\"k2 p8: exact-size cudaHostAlloc host buffer (host VA == device alias, also on WSL2).\"\"\"\n"
    "    if not pin_memory:\n"
    "        return allocator.allocate(dims, dtype=dtype, device=device)\n"
    "    import time as _time\n"
    "\n"
    "    from sglang.srt.mem_cache.pool_host.exact_pinned import exact_pinned_empty\n"
    "\n"
    "    t0 = _time.perf_counter()\n"
    "    buffer = exact_pinned_empty(dims, dtype)\n"
    "    # Nothing was cudaHostRegister-ed, so host-pool destroy() must not unregister it.\n"
    "    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, [])\n"
    "    logger.info(\n"
    "        \"HiCache host buffer %s %s: %.2f GB via exact cudaHostAlloc in %.1f s\",\n"
    "        tuple(dims),\n"
    "        dtype,\n"
    "        buffer.numel() * buffer.element_size() / 1e9,\n"
    "        _time.perf_counter() - t0,\n"
    "    )\n"
    "    return buffer\n"
    "\n"
    "\n"
    "@lru_cache(maxsize=1)\ndef _resolve_device_accessible_ptr_fn():\n",
    "insert alloc_with_cuda_host_alloc",
)
t = replace_once(
    t,
    "ALLOC_MEMORY_FUNCS = defaultdict(\n"
    "    lambda: alloc_with_host_register,\n"
    "    {\n"
    "        \"npu\": alloc_with_pin_memory,\n"
    "        \"musa\": alloc_with_pin_memory,\n"
    "    },\n"
    ")\n",
    "ALLOC_MEMORY_FUNCS = defaultdict(\n"
    "    lambda: alloc_with_host_register,\n"
    "    {\n"
    "        \"npu\": alloc_with_pin_memory,\n"
    "        \"musa\": alloc_with_pin_memory,\n"
    "    },\n"
    ")\n"
    "\n"
    "# k2 p8 (env SGLANG_HICACHE_HOST_ALLOC=cuda, default off): on WSL2 mmap + cudaHostRegister memory has a\n"
    "# device alias != host VA and this sgl_kernel cannot translate, so allocate host pools with cudaHostAlloc.\n"
    "if os.environ.get(\"SGLANG_HICACHE_HOST_ALLOC\", \"\").strip().lower() == \"cuda\":\n"
    "    ALLOC_MEMORY_FUNCS.default_factory = lambda: alloc_with_cuda_host_alloc\n",
    "ALLOC_MEMORY_FUNCS default",
)
COMMON.write_text(t)
print("p8_hicache_cuda_host_alloc applied")
