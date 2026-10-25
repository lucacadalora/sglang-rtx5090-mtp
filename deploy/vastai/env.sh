# Sourced by the scripts in this folder (not run on its own).
#
# vast.ai SSH sessions do not always inherit the image's ENV (the PATH to SGLang's venv in /opt/sglang/bin,
# CUDA_HOME, the library paths that Triton and FlashInfer need to JIT-compile kernels). PID 1 was started by Docker
# with the image environment, so take those variables from it, then fall back to the values baked into
# lmsysorg/sglang:v0.5.20-cu130.
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in
      PATH=*|LD_LIBRARY_PATH=*|LIBRARY_PATH=*|CUDA_HOME=*|LANG=*|LC_ALL=*|FLASHINFER_VERSION=*|SGLANG_RUST_BUILD_MODE=*)
        export "${kv?}" ;;
    esac
  done < /proc/1/environ
fi
case ":$PATH:" in
  *:/opt/sglang/bin:*) ;;
  *) export PATH="/opt/sglang/bin:/usr/local/cuda/bin:$PATH" ;;
esac
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LIBRARY_PATH=${LIBRARY_PATH:-/usr/local/cuda/lib64/stubs}
case ":${LD_LIBRARY_PATH:-}:" in
  *:/usr/local/cuda/lib64:*) ;;
  *) export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export LANG=${LANG:-en_US.UTF-8} LC_ALL=${LC_ALL:-en_US.UTF-8}

# Where things live (override before running a script). /workspace is on the instance's own disk.
export SERVE_ROOT=${SERVE_ROOT:-/workspace/sglang}   # API keys, chat template, logs, kernel caches
export HF_HOME=${HF_HOME:-/workspace/hf}         # model weights, Hugging Face cache layout
SGL_SRC=/sgl-workspace/sglang                    # the image's SGLang source tree (installed in editable mode)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Pinned weights: the exact revisions measured on our card.
GT_REPO=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090    # all-NVFP4 27B, ships without its MTP layer
GT_REV=5b7a687fc8211a5d631c8ca6a593dd37eb26ce33
RX_REPO=RadixArk/Qwen3.8-27B-NVFP4                       # source of the bf16 MTP head: index + shard 3 only (2 GB)
RX_REV=319f741cce68d7914884900c138a1fbb70a42f30
MOE_REPO=nvidia/Qwen3.6-35B-A3B-NVFP4
MOE_REV=1355db6a052410cfd62085d94b58866fd0f2c3c5
DF_NVFP4_REPO=maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal  # DFlash2 drafter, community NVFP4 conversion (1.55 GB)
DF_NVFP4_REV=bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
DF_BF16_REPO=z-lab/Qwen3.8-27B-DFlash2                   # DFlash2 drafter, first-party bf16 (3.85 GB)
DF_BF16_REV=50307d4c4cde6860d4eee73e2547cd786fe8e8a4
TPL_REPO=Qwen/Qwen3.8-27B                                # official chat template (the one we served)
TPL_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
TPL_SHA256=c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041

snapshot_dir() { echo "$HF_HOME/hub/models--${1//\//--}/snapshots/$2"; }
