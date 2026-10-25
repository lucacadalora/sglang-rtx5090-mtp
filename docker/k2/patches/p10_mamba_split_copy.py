#!/usr/bin/env python3
"""k2 p10: split GDN state rows for HiCache transfers (env SGLANG_HICACHE_MAMBA_SPLIT_BYTES=<n>, default 0 = off).

MambaPoolHost._copy_tensor (kernel io backend) reuses transfer_kv_per_layer_mla, which copies one index per warp.
A GDN layer's state for one slot is ~1.5 MiB, so a one-slot backup or restore moves at ~0.4 GB/s: measured
184 ms per slot (48 layers) on this box, and the restore sits on the TTFT critical path of every host hit.
With the env var set, each row is viewed as rows of n bytes (n must divide the row size) and the indices are
expanded to match, so many warps share one state. Same bytes, same stream, same kernel; rows that n does not
divide, and non-contiguous tensors, take the original path.
"""
import pathlib

F = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/mamba.py")


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p10 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


t = F.read_text()
assert "_K2_MAMBA_SPLIT_BYTES" not in t, "p10 already applied"
t = replace_once(t, "import logging\nimport threading\n", "import logging\nimport os\nimport threading\n", "import os")
t = replace_once(
    t,
    "logger = logging.getLogger(__name__)\n",
    "logger = logging.getLogger(__name__)\n"
    "\n"
    "# k2 p10 (env SGLANG_HICACHE_MAMBA_SPLIT_BYTES, default 0 = off): sub-row size for GDN state transfers.\n"
    "_K2_MAMBA_SPLIT_BYTES = int(os.environ.get(\"SGLANG_HICACHE_MAMBA_SPLIT_BYTES\", \"0\") or 0)\n",
    "split constant",
)
t = replace_once(
    t,
    "        if io_backend == \"kernel\":\n"
    "            # TODO: Rename the interface for clarity.\n"
    "            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.\n"
    "            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.\n"
    "            transfer_kv_per_layer_mla(\n"
    "                src=src,\n"
    "                dst=dst,\n"
    "                src_indices=src_indices,\n"
    "                dst_indices=dst_indices,\n"
    "                item_size=MambaPoolHost._item_size_per_index(src),\n"
    "            )\n",
    "        if io_backend == \"kernel\":\n"
    "            # TODO: Rename the interface for clarity.\n"
    "            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.\n"
    "            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.\n"
    "            item_size = MambaPoolHost._item_size_per_index(src)\n"
    "            # k2 p10: the kernel copies one index per warp; split each state row so many warps share it.\n"
    "            split = _K2_MAMBA_SPLIT_BYTES\n"
    "            if (\n"
    "                split > 0\n"
    "                and item_size > split\n"
    "                and item_size % split == 0\n"
    "                and src.is_contiguous()\n"
    "                and dst.is_contiguous()\n"
    "            ):\n"
    "                k = item_size // split\n"
    "                ar = torch.arange(k, device=src_indices.device, dtype=src_indices.dtype)\n"
    "                transfer_kv_per_layer_mla(\n"
    "                    src=src.view(torch.uint8).view(-1, split),\n"
    "                    dst=dst.view(torch.uint8).view(-1, split),\n"
    "                    src_indices=(src_indices.unsqueeze(1) * k + ar).reshape(-1),\n"
    "                    dst_indices=(dst_indices.unsqueeze(1) * k + ar).reshape(-1),\n"
    "                    item_size=split,\n"
    "                )\n"
    "                return\n"
    "            transfer_kv_per_layer_mla(\n"
    "                src=src,\n"
    "                dst=dst,\n"
    "                src_indices=src_indices,\n"
    "                dst_indices=dst_indices,\n"
    "                item_size=item_size,\n"
    "            )\n",
    "_copy_tensor kernel branch",
)
F.write_text(t)
print("p10_mamba_split_copy applied")
