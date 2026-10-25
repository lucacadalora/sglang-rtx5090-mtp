#!/usr/bin/env python3
"""k2 p12: HiCache host-memory reserve in GiB (env SGLANG_HICACHE_HOST_RESERVE_GB, default 10 = upstream).

pool_host/base.py keeps HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 GiB of host RAM free when it sizes a HiCache pool
(host_memory_budget_bytes() = MemAvailable - reserve). Under WSL2 the VM's RAM is capped by .wslconfig and the
server is the VM's main tenant, so 10 GiB of permanently idle headroom is cache we could hold instead. With the env
var set the reserve is that many GiB; unset keeps upstream behaviour exactly.
"""
import pathlib

F = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/base.py")


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p12 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


t = F.read_text()
assert "SGLANG_HICACHE_HOST_RESERVE_GB" not in t, "p12 already applied"
t = replace_once(t, "import abc\nimport logging\nimport threading\n", "import abc\nimport logging\nimport os\nimport threading\n",
                 "import os")
t = replace_once(
    t,
    "# Host RAM to leave free when sizing HiCache pools (OS, other processes).\n"
    "HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)\n",
    "# Host RAM to leave free when sizing HiCache pools (OS, other processes).\n"
    "# k2 p12 (env SGLANG_HICACHE_HOST_RESERVE_GB, default 10 = upstream): the reserve in GiB. Under WSL2 the VM's RAM\n"
    "# is capped by .wslconfig, so a smaller reserve turns idle VM headroom into cache.\n"
    "HICACHE_HOST_MEMORY_RESERVE_BYTES: int = int(\n"
    "    float(os.environ.get(\"SGLANG_HICACHE_HOST_RESERVE_GB\", \"10\") or 10) * (1024**3)\n"
    ")\n",
    "reserve constant",
)
F.write_text(t)
print("p12_host_reserve applied")
