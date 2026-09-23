#!/usr/bin/env python3
"""Build a local 'gittensor NVFP4 + MTP head' checkpoint from two snapshots already in the HF cache.

The gittensor NVFP4 checkpoint (gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 @5b7a687) ships without the model's
multi-token-prediction (MTP) layer and declares mtp_num_hidden_layers = 0. Its bf16 mtp.* tensors (15 tensors,
849,398,784 bytes) are byte-identical in Qwen/Qwen3.8-27B, RadixArk/Qwen3.8-27B-NVFP4 and nvidia/Qwen3.8-27B-NVFP4
(checked with HF range reads), so this script copies them from a RadixArk snapshot, symlinks every other gittensor
file, extends the weight index and sets mtp_num_hidden_layers = 1. Nothing is downloaded; +0.85 GB on disk.

Run in a throwaway CPU-only container (no --gpus), from the repo root:
  docker run --rm -v "${SGL_ROOT:-/opt/sglang}/huggingface:/root/.cache/huggingface" \
     -v "$PWD/scripts:/scripts:ro" --entrypoint python3 lmsysorg/sglang:v0.5.20-cu130 /scripts/graft_mtp.py
Output: $SGL_ROOT/huggingface/local/gittensor-mtp (the path scripts/launch.sh expects).
Override the snapshots with GT_SNAPSHOT / MTP_SOURCE_SNAPSHOT (paths relative to the HF cache root).
"""
import json, os
from safetensors import safe_open
from safetensors.torch import save_file

HF = "/root/.cache/huggingface"
GT = os.environ.get("GT_SNAPSHOT",
                    "hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33")
RX = os.environ.get("MTP_SOURCE_SNAPSHOT",
                    "hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30")
OUT = os.path.join(HF, "local/gittensor-mtp")

os.makedirs(OUT, exist_ok=True)
gt_dir, rx_dir = os.path.join(HF, GT), os.path.join(HF, RX)

# 1) symlink every gittensor file except the two we rewrite (relative links work on the host and in the container)
for f in os.listdir(gt_dir):
    if f in ("config.json", "model.safetensors.index.json"):
        continue
    dst = os.path.join(OUT, f)
    if not os.path.lexists(dst):
        os.symlink(os.path.relpath(os.path.join(gt_dir, f), OUT), dst)

# 2) extract the bf16 MTP head from the source snapshot's shards
rx_idx = json.load(open(os.path.join(rx_dir, "model.safetensors.index.json")))["weight_map"]
mtp = {}
for name, shard in rx_idx.items():
    if name.startswith("mtp."):
        with safe_open(os.path.join(rx_dir, shard), framework="pt") as fh:
            mtp[name] = fh.get_tensor(name).contiguous()
nbytes = sum(t.numel() * t.element_size() for t in mtp.values())
assert len(mtp) == 15 and nbytes == 849398784, (len(mtp), nbytes)
save_file(mtp, os.path.join(OUT, "mtp.safetensors"), metadata={"format": "pt"})

# 3) index = gittensor index + mtp.* -> mtp.safetensors
idx = json.load(open(os.path.join(gt_dir, "model.safetensors.index.json")))
for name in mtp:
    idx["weight_map"][name] = "mtp.safetensors"
idx.setdefault("metadata", {})["total_size"] = int(idx["metadata"].get("total_size", 0)) + nbytes
json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

# 4) config: declare one MTP layer (gittensor ships mtp_num_hidden_layers = 0)
cfg = json.load(open(os.path.join(gt_dir, "config.json")))
for c in (cfg, cfg.get("text_config", {})):
    if "mtp_num_hidden_layers" in c:
        c["mtp_num_hidden_layers"] = 1
cfg.setdefault("text_config", {})["mtp_num_hidden_layers"] = 1
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print("OK", OUT, len(mtp), "tensors", nbytes, "bytes")
