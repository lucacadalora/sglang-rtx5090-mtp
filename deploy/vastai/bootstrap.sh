#!/usr/bin/env bash
# One-time setup of a rented RTX 5090 running the image lmsysorg/sglang:v0.5.20-cu130 (vast.ai, or any Linux box or
# container with that image). Run as root inside the instance, from a clone of this repo:
#   bash deploy/vastai/bootstrap.sh                       # weights for the 27B with MTP (default)
#   MODELS="27b-mtp moe" bash deploy/vastai/bootstrap.sh  # both models (~45 GB of downloads)
#   MODELS=moe bash deploy/vastai/bootstrap.sh            # the 35B-A3B MoE only
#   MODELS="27b-mtp 27b-dflash" bash deploy/vastai/bootstrap.sh   # + the DFlash2 drafters for the A/B (~5.4 GB;
#                                                         # DFLASH_DRAFTS="nvfp4 bf16" by default, or just one)
# Steps, each skipped when it is already done, so re-running is safe:
#   1. preflight  sm_120 GPU (RTX 5090 class), driver new enough for the CUDA 13 image, SGLang v0.5.20, free disk
#   2. patches    SGLang in the image is patched in place, in the same order as the Docker layers in docker/:
#                 mtpfix (PR #37155) -> embhost (PR #37826 backport) -> k1 -> k2. Every k1/k2 change is env-gated
#                 and stays off unless serve.sh turns it on.
#   3. weights    pinned revisions (env.sh): 27b-mtp ~19 GB, moe ~24 GB, 27b-dflash drafters 1.55 + 3.85 GB
#   4. MTP graft  gittensor NVFP4 + the bf16 MTP head from RadixArk -> $HF_HOME/local/gittensor-mtp
#   5. template   Qwen's official Qwen3.8-27B chat template (sha256 checked)
#   6. API keys   $SERVE_ROOT/api_key and $SERVE_ROOT/admin_api_key (mode 600)
# Test the patch step without a GPU or downloads: SKIP_GPU_CHECK=1 MODELS=none bash deploy/vastai/bootstrap.sh
set -euo pipefail
# shellcheck source=deploy/vastai/env.sh
source "$(dirname "$0")/env.sh"
MODELS=${MODELS:-27b-mtp}
DFLASH_DRAFTS=${DFLASH_DRAFTS:-nvfp4 bf16}
T0=$(date +%s)
log() { printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }
want() { [[ " $MODELS " == *" $1 "* ]]; }

for m in $MODELS; do
  case "$m" in 27b-mtp|27b|27b-dflash|moe|none) ;; *) die "unknown entry '$m' in MODELS (27b-mtp | 27b | 27b-dflash | moe | none)" ;; esac
done
for d in $DFLASH_DRAFTS; do
  case "$d" in nvfp4|bf16) ;; *) die "unknown entry '$d' in DFLASH_DRAFTS (nvfp4 | bf16)" ;; esac
done
draft_ref() {  # nvfp4|bf16 -> "repo revision"
  if [ "$1" = nvfp4 ]; then echo "$DF_NVFP4_REPO $DF_NVFP4_REV"; else echo "$DF_BF16_REPO $DF_BF16_REV"; fi
}
mkdir -p "$SERVE_ROOT/templates" "$SERVE_ROOT/logs" "$SERVE_ROOT/cache" "$HF_HOME"
chmod 700 "$SERVE_ROOT"

# ---------------------------------------------------------------------------------------------------------- 1
log "preflight"
[ "$(id -u)" = 0 ] || die "run as root (vast.ai SSH logins are root)"
[ -d "$SGL_SRC/python/sglang" ] || die "$SGL_SRC not found: create the instance from the image lmsysorg/sglang:v0.5.20-cu130"
VER=$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null) || die "python3 cannot import sglang (PATH=$PATH)"
[ "$VER" = 0.5.20 ] || die "found sglang $VER; the patches anchor on v0.5.20 source text (image lmsysorg/sglang:v0.5.20-cu130)"
echo "SGLang $VER at $SGL_SRC"
if [ "${SKIP_GPU_CHECK:-0}" != 1 ]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this container has no GPU"
  IFS=, read -r GPU_NAME CC MEM DRV < <(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv,noheader,nounits | head -1)
  CC=${CC// /}; DRV=${DRV// /}
  echo "GPU: ${GPU_NAME# } (sm ${CC}), ${MEM// /} MiB, driver ${DRV}"
  [ "$CC" = 12.0 ] || die "need an RTX 5090-class Blackwell GPU (compute capability 12.0), found $CC. The NVFP4 weights and the k1 kernels are sm_120 work; an H100/H200 needs a different recipe"
  [ "${DRV%%.*}" -ge 580 ] || die "driver $DRV is too old for the CUDA 13 image (580 or newer): rent an offer with Max CUDA >= 13.0"
  [ "$(nvidia-smi -L | wc -l)" -eq 1 ] || echo "note: more than one GPU is visible; serve.sh uses GPU 0 only"
fi
MEM_GB=$(awk '/^MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
case "$LIM" in max|'') LIM_TXT="no container limit" ;; *) LIM_TXT="container limit $((LIM / 1073741824)) GiB" ;; esac
echo "RAM: ${MEM_GB} GiB on the host, ${LIM_TXT}"
SHM_MB=$(df -BM --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)
[ "${SHM_MB:-0}" -ge 1024 ] || echo "note: /dev/shm is only ${SHM_MB} MB; one-GPU serving normally copes, but add --shm-size=8g to the instance's Docker options if the server fails at start"
NEED_GB=0
if want 27b-mtp || want 27b || want 27b-dflash; then
  [ -f "$(snapshot_dir "$GT_REPO" "$GT_REV")/model-00002-of-00002.safetensors" ] || NEED_GB=$((NEED_GB + 20))
fi
if want 27b-dflash; then
  for d in $DFLASH_DRAFTS; do
    read -r DR DV < <(draft_ref "$d")
    [ -s "$(snapshot_dir "$DR" "$DV")/model.safetensors" ] || NEED_GB=$((NEED_GB + 4))
  done
fi
if want moe; then
  [ -f "$(snapshot_dir "$MOE_REPO" "$MOE_REV")/model-00003-of-00003.safetensors" ] || NEED_GB=$((NEED_GB + 25))
fi
FREE_GB=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc 0-9)
echo "disk: ${FREE_GB} GB free under $HF_HOME, ~${NEED_GB} GB still to download"
[ "$FREE_GB" -ge $((NEED_GB + 5)) ] || die "not enough disk; rent the instance with more disk space (README: 150 GB or more)"

# ---------------------------------------------------------------------------------------------------------- 2
log "patches (in place under $SGL_SRC)"
STAMP=$SGL_SRC/.rtx5090-patches
FINGERPRINT=$(cat "$REPO"/docker/mtpfix/pr37155.diff "$REPO"/docker/embhost/*.py "$REPO"/docker/k1/patches/*.py \
  "$REPO"/docker/k2/patches/*.py | sha256sum | cut -c1-16)
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$FINGERPRINT" ]; then
  echo "already applied (fingerprint $FINGERPRINT)"
else
  for f in "$REPO"/docker/mtpfix/pr37155.diff "$REPO"/docker/embhost/*.py "$REPO"/docker/k1/*.py \
           "$REPO"/docker/k1/patches/*.py "$REPO"/docker/k2/*.py "$REPO"/docker/k2/patches/*.py; do
    if grep -q $'\r' "$f"; then die "$f has CRLF line endings (clone with: git -c core.autocrlf=false clone ...)"; fi
  done
  # Start from the image's pristine tree: an earlier run that stopped half-way would fail the anchor checks.
  git -C "$SGL_SRC" checkout -q -- python
  git -C "$SGL_SRC" clean -qfd python
  rm -f "$STAMP"
  (cd "$SGL_SRC" && patch -p1 --quiet < "$REPO/docker/mtpfix/pr37155.diff")
  python3 -c "import sglang.srt.speculative.base_spec_worker as m; assert hasattr(m.EagleDraftWorkerBase, 'share_target_embed_and_head')"
  echo "mtpfix: PR #37155 applied (MTP draft shares the embedding and LM head before the KV pool is sized)"
  cp "$REPO/docker/embhost/host_embedding_gather.py" /tmp/host_embedding_gather.py
  python3 "$REPO/docker/embhost/apply_embed_offload.py"
  python3 -c "import sglang.srt.layers.vocab_parallel_embedding as v; assert hasattr(v, 'offload_input_embeddings_to_host'); import sglang.kernels.ops.embeddings.host_embedding_gather"
  python3 -m py_compile "$SGL_SRC/python/sglang/srt/model_executor/model_runner.py"
  rm -f /tmp/host_embedding_gather.py
  echo "embhost: PR #37826 backport applied (input embedding table in pinned host RAM)"
  touch /tmp/k1_stamp
  sleep 1
  for p in p1_ba_tiny_gemm p2_kv_unit_scale p3_lmhead_autotune p7_silu_fp4_attn_mlp p6_syncfree_seam; do
    python3 "$REPO/docker/k1/patches/$p.py"
  done
  python3 "$REPO/docker/k1/k1_check.py" /tmp/k1_stamp
  rm -f /tmp/k1_stamp
  echo "k1: kernel/runtime patches applied (env-gated)"
  for p in p8_hicache_cuda_host_alloc p9_embed_exact_pin p10_mamba_split_copy p11_no_staged_write_back p12_host_reserve; do
    python3 "$REPO/docker/k2/patches/$p.py"
  done
  python3 "$REPO/docker/k2/k2_check.py"
  echo "k2: HiCache patches applied (env-gated)"
  echo "$FINGERPRINT" > "$STAMP"
fi

# ---------------------------------------------------------------------------------------------------------- 3
fetch() {  # repo revision [file patterns...]
  python3 - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, *patterns = sys.argv[1:]
path = snapshot_download(repo_id=repo, revision=rev, allow_patterns=patterns or None, max_workers=16)
print(f"   {repo} @{rev[:7]} -> {path}")
PY
}
complete() {  # snapshot_dir [file ...]: every listed file present, and every shard in its weight index
  python3 - "$@" <<'PY'
import json, os, sys
d, *need = sys.argv[1:]
idx = os.path.join(d, "model.safetensors.index.json")
if not need:
    need = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    if os.path.exists(idx):
        need += sorted(set(json.load(open(idx))["weight_map"].values()))
sys.exit(0 if all(os.path.exists(os.path.join(d, f)) for f in need) else 1)
PY
}
if want 27b-mtp || want 27b || want 27b-dflash; then
  log "weights: $GT_REPO"
  if complete "$(snapshot_dir "$GT_REPO" "$GT_REV")"; then echo "already complete"; else fetch "$GT_REPO" "$GT_REV"; fi
fi
if want 27b-mtp; then
  log "MTP head source: $RX_REPO (index + shard 3 only)"
  if complete "$(snapshot_dir "$RX_REPO" "$RX_REV")" model.safetensors.index.json model-00003-of-00003.safetensors; then
    echo "already complete"
  else
    fetch "$RX_REPO" "$RX_REV" model.safetensors.index.json model-00003-of-00003.safetensors
  fi
fi
if want moe; then
  log "weights: $MOE_REPO"
  if complete "$(snapshot_dir "$MOE_REPO" "$MOE_REV")"; then echo "already complete"; else fetch "$MOE_REPO" "$MOE_REV"; fi
fi
if want 27b-dflash; then
  for d in $DFLASH_DRAFTS; do
    read -r DR DV < <(draft_ref "$d")
    log "DFlash2 drafter ($d): $DR"
    # weights and configs only; nothing from the repo is executed (no trust_remote_code)
    if complete "$(snapshot_dir "$DR" "$DV")" config.json model.safetensors; then
      echo "already complete"
    else
      fetch "$DR" "$DV" "*.json" "*.safetensors"
    fi
  done
fi

# ---------------------------------------------------------------------------------------------------------- 4
if want 27b-mtp; then
  log "MTP graft"
  OUT=$HF_HOME/local/gittensor-mtp
  if [ -s "$OUT/mtp.safetensors" ] && grep -q '"mtp_num_hidden_layers": 1' "$OUT/config.json" 2>/dev/null; then
    echo "already grafted: $OUT"
  else
    HF_CACHE_ROOT=$HF_HOME python3 "$REPO/scripts/graft_mtp.py"
  fi
fi

# ---------------------------------------------------------------------------------------------------------- 5
if want 27b-mtp || want 27b || want 27b-dflash; then
  log "chat template"
  TPL=$SERVE_ROOT/templates/qwen3.8-upstream.jinja
  if echo "$TPL_SHA256  $TPL" | sha256sum -c --status 2>/dev/null; then
    echo "already in place: $TPL"
  else
    SRC=$(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('$TPL_REPO', 'chat_template.jinja', revision='$TPL_REV'))")
    cp "$SRC" "$TPL"
    echo "$TPL_SHA256  $TPL" | sha256sum -c --quiet || die "chat template hash mismatch ($TPL)"
    echo "saved $TPL (sha256 verified)"
  fi
fi

# ---------------------------------------------------------------------------------------------------------- 6
log "API keys"
if [ -s "$SERVE_ROOT/api_key" ] && [ -s "$SERVE_ROOT/admin_api_key" ]; then
  echo "kept the existing keys in $SERVE_ROOT"
else
  (umask 077; openssl rand -hex 24 > "$SERVE_ROOT/api_key"; openssl rand -hex 24 > "$SERVE_ROOT/admin_api_key")
  echo "created $SERVE_ROOT/api_key (clients) and $SERVE_ROOT/admin_api_key"
fi

log "done in $(( $(date +%s) - T0 )) s"
echo "next: bash deploy/vastai/serve.sh           (PROFILE=moe for the MoE, PROFILE=27b-dflash for DFlash2)"
