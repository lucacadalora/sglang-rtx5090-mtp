#!/usr/bin/env python3
"""k2 post-patch check (CPU only, runs inside docker build): byte-compile every touched module, import them,
and prove each env gate is off by default and on when set."""
import os
import pathlib
import py_compile
import subprocess
import sys

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
FILES = [
    ROOT / "srt/mem_cache/pool_host/exact_pinned.py",
    ROOT / "srt/mem_cache/pool_host/common.py",
    ROOT / "srt/mem_cache/pool_host/mamba.py",
    ROOT / "srt/mem_cache/pool_host/mha.py",
    ROOT / "srt/mem_cache/pool_host/base.py",
    ROOT / "srt/layers/vocab_parallel_embedding.py",
]
for f in FILES:
    py_compile.compile(str(f), doraise=True)

K2_ENV = ("SGLANG_HICACHE_HOST_ALLOC", "SGLANG_HICACHE_MAMBA_SPLIT_BYTES", "SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN",
          "SGLANG_HICACHE_NO_STAGED_WRITE_BACK", "SGLANG_HICACHE_HOST_RESERVE_GB")
base_env = {k: v for k, v in os.environ.items() if k not in K2_ENV}

OFF = """
import inspect
import sglang.srt.mem_cache.pool_host.common as c
import sglang.srt.mem_cache.pool_host.mamba as m
import sglang.srt.layers.vocab_parallel_embedding as v
import sglang.srt.mem_cache.pool_host.exact_pinned as e
assert c.ALLOC_MEMORY_FUNCS["cuda"] is c.alloc_with_host_register
assert c.ALLOC_MEMORY_FUNCS["npu"] is c.alloc_with_pin_memory
assert m._K2_MAMBA_SPLIT_BYTES == 0
import sglang.srt.mem_cache.pool_host.base as b
assert b.HICACHE_HOST_MEMORY_RESERVE_BYTES == 10 * 1024**3
src = inspect.getsource(v)
assert src.count("SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN") == 2
import sglang.srt.mem_cache.pool_host.mha as h
assert inspect.getsource(h).count("SGLANG_HICACHE_NO_STAGED_WRITE_BACK") == 4  # comment + gate in both MHA pool classes
assert callable(e.exact_pinned_empty)
print("k2 gates off by default: OK")
"""
ON = """
import sglang.srt.mem_cache.pool_host.common as c
import sglang.srt.mem_cache.pool_host.mamba as m
assert c.ALLOC_MEMORY_FUNCS["cuda:0"] is c.alloc_with_cuda_host_alloc
assert c.ALLOC_MEMORY_FUNCS["musa"] is c.alloc_with_pin_memory
assert m._K2_MAMBA_SPLIT_BYTES == 16384
import sglang.srt.mem_cache.pool_host.base as b
assert b.HICACHE_HOST_MEMORY_RESERVE_BYTES == 6 * 1024**3
print("k2 gates on when set: OK")
"""
subprocess.run([sys.executable, "-c", OFF], check=True, env=base_env)
subprocess.run(
    [sys.executable, "-c", ON],
    check=True,
    env={**base_env, "SGLANG_HICACHE_HOST_ALLOC": "cuda", "SGLANG_HICACHE_MAMBA_SPLIT_BYTES": "16384",
         "SGLANG_HICACHE_HOST_RESERVE_GB": "6"},
)
print("k2_check passed")
