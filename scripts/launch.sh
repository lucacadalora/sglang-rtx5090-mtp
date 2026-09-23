#!/usr/bin/env bash
# Launch Qwen3.8-27B NVFP4 with MTP speculative decoding (or the Qwen3.6-35B-A3B MoE) on one RTX 5090 under WSL2.
# Run as root inside the WSL distro (needs docker with the NVIDIA runtime, and nvidia-smi):
#   bash scripts/launch.sh                  # PROFILE=27b-mtp
#   PROFILE=27b bash scripts/launch.sh      # same weights, no MTP, stock image (the rollback)
#   PROFILE=moe bash scripts/launch.sh
#   DRY_RUN=1 MEM_FRACTION=0.773 bash scripts/launch.sh   # print the sglang arguments, touch nothing
#
# PROFILE:
#   27b-mtp  gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 @5b7a687 + the grafted bf16 MTP head (scripts/graft_mtp.py),
#            image local/sglang:v0.5.20-cu130-mtpfix-embhost (docker/build.sh): EAGLE 3 steps / top-k 1 / 4 draft tokens
#            with ReplaySSM verification, input embedding in pinned host RAM, 147,456-token window, 5 running.
#            Falls back to the plain 27B with a WARNING if the graft or the image is missing.
#   27b      the same weights without MTP on the stock image.
#   moe      nvidia/Qwen3.6-35B-A3B-NVFP4 @1355db6a on the stock image, 131,072-token window, 8 running.
#
# Everything lives under SGL_ROOT (default /opt/sglang): huggingface/ (HF cache, mounted into the container),
# sglang-cache/, templates/, and api_key + admin_api_key (generated on first run, mode 600). The server listens on
# 127.0.0.1:${HOST_PORT:-30000} only and requires the API key.
#
# WSL2 memory: CUDA under WSL2 sees ~30 GiB and ignores the VRAM that Windows desktop apps hold, and it never reports
# out-of-memory. When the card is oversubscribed, WDDM pages model memory to system RAM and every decode step re-reads
# it over PCIe: measured 96 -> ~15 tok/s, then stuck at ~75 until a restart. So this script measures the desktop's VRAM
# at launch, sizes --mem-fraction-static to leave HEADROOM_MIB free, and the container returns PyTorch's cached blocks
# when idle (SGLANG_EMPTY_CACHE_INTERVAL). Watch for it with:  nvidia-smi dmon -s pmt  (GB/s of PCIe during decode
# plus < 0.5 GB free = paging; healthy decode moves 15-80 MB/s).
#
# Knobs (env vars): PROFILE, MEM_FRACTION (overrides auto-sizing; pass it with DRY_RUN=1 while a server is running,
# or its VRAM is counted as desktop use), HEADROOM_MIB, NONSTATIC_MIB, DESKTOP_MIN_MIB (assume at least this much
# desktop VRAM, e.g. at Windows logon before apps open), CONTEXT_LENGTH, MAX_RUNNING, MAMBA_SLOTS, KV_DTYPE, IMAGE,
# HOST_PORT, CONTAINER_NAME, EXTRA_ARGS.
set -euo pipefail

ROOT=${SGL_ROOT:-/opt/sglang}
NAME=${CONTAINER_NAME:-sgl-5090}
IMAGE_DEFAULT=lmsysorg/sglang:v0.5.20-cu130
PROFILE=${PROFILE:-27b-mtp}
GT_SNAPSHOT=huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33
GT_MTP_DIR=huggingface/local/gittensor-mtp
MOE_SNAPSHOT=huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
MTP_IMAGE=local/sglang:v0.5.20-cu130-mtpfix-embhost
# Optional: Qwen/Qwen3.8-27B's official chat_template.jinja saved under this name. The gittensor repo ships its own
# rewritten template; with this file present the official one is used instead (that is what we served).
TEMPLATE=templates/qwen3.8-upstream.jinja

PROFILE_HEADROOM=1792      # free VRAM kept for desktop swings (2.3-4.5 GB during a day) and allocator spikes
PROFILE_NONSTATIC=2600     # CUDA context + workspace + decode graphs + activations, measured on the 27B
PROFILE_SKIP_LOCK=0
PROFILE_OFFLOAD_EMBEDDING=0
PROFILE_IMAGE=""

case "$PROFILE" in
  27b-mtp|27b)
    USE_MTP=0
    if [ "$PROFILE" = 27b-mtp ]; then
      if [ ! -s "$ROOT/$GT_MTP_DIR/mtp.safetensors" ]; then
        echo "WARNING: $ROOT/$GT_MTP_DIR is missing (run scripts/graft_mtp.py); serving the 27B without MTP"
      elif [ "${DRY_RUN:-0}" != 1 ] && ! docker image inspect "$MTP_IMAGE" >/dev/null 2>&1; then
        echo "WARNING: image $MTP_IMAGE is missing (run docker/build.sh); serving the 27B without MTP"
      else
        USE_MTP=1
      fi
    fi
    MODEL=/root/.cache/$GT_SNAPSHOT
    SERVED_NAME=qwen3.8-27b
    WEIGHT_MIB=${WEIGHT_MIB:-17510}        # loaded weights (boot log: 'mem usage=17.10 GB')
    KV_KIB_FP8=32                          # 16 full-attention layers x 4 KV heads x 256 x (K+V) x 1 B
    KV_HEADS=4
    SLOT_MIB=74.8125                       # one bf16 Gated DeltaNet state slot as sized by v0.5.20
    PROFILE_MAX_RUNNING=5
    MAMBA_SLOTS=${MAMBA_SLOTS:-22}
    WANT_CONTEXT=${CONTEXT_LENGTH:-147456} # 131,072 input + 16,384 output
    # State slots: extra_buffer_lazy gives a running request 1 ping-pong slot instead of 2, so v0.5.20 caps running
    # requests at slots/4 (22 slots -> 5). The path cap keeps the 2 deepest saved states per conversation (prompt end
    # + reply end), which is all the next turn needs; eviction is least-recently-used.
    PROFILE_ARGS="--mamba-radix-cache-strategy extra_buffer_lazy --mamba-max-states-per-path 2 --sleep-on-idle"
    if [ -s "$ROOT/$TEMPLATE" ]; then PROFILE_ARGS="--chat-template /templates/qwen3.8-upstream.jinja $PROFILE_ARGS"; fi
    if [ "$USE_MTP" = 1 ]; then
      # Image = v0.5.20 + PR #37155 (the draft shares the target's embedding and LM head before the KV pool is sized)
      # + PR #37826 backport (SGLANG_OFFLOAD_EMBEDDING_TO_HOST: the 2.37 GiB input-embedding table is read from pinned
      # host RAM through UVA; bit-exact, 12 us per decode step). Measured mean accept length 2.57.
      MODEL=/root/.cache/$GT_MTP_DIR
      PROFILE_IMAGE=$MTP_IMAGE
      PROFILE_OFFLOAD_EMBEDDING=1
      WEIGHT_MIB=15700                     # target without the embedding + bf16 MTP layer + ReplaySSM ring (within 0.4%)
      KV_KIB_FP8=34                        # EAGLE adds the MTP layer's own KV
      PROFILE_NONSTATIC=3000               # draft/verify graphs and buffers
      PROFILE_ARGS="$PROFILE_ARGS --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec"
    fi
    if [ "${DRY_RUN:-0}" != 1 ]; then
      [ -f "$ROOT/$GT_SNAPSHOT/model-00002-of-00002.safetensors" ] || { echo "error: weights missing under $ROOT/$GT_SNAPSHOT"; exit 1; }
    fi
    ;;
  moe)
    MODEL=/root/.cache/$MOE_SNAPSHOT
    SERVED_NAME=qwen3.6-35b-a3b
    WEIGHT_MIB=${WEIGHT_MIB:-21170}        # boot log: 'mem usage=20.67 GB' (the unused MTP layer is not loaded)
    KV_KIB_FP8=10                          # 10 full-attention layers x 2 KV heads x 256 x (K+V) x 1 B
    KV_HEADS=2
    SLOT_MIB=31.3
    PROFILE_MAX_RUNNING=8                  # with SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 a running request pins 3 slots, not 4
    MAMBA_SLOTS=${MAMBA_SLOTS:-32}
    WANT_CONTEXT=${CONTEXT_LENGTH:-131072}
    PROFILE_SKIP_LOCK=1
    # NVFP4 MoE on the 5090 (sm_120): v0.5.20 auto-selects the SM100-only flashinfer_trtllm runner, which dies with
    # NotImplementedError before the first token; flashinfer_cutlass works (open PR #39807 makes it the default).
    PROFILE_ARGS="--moe-runner-backend flashinfer_cutlass --mamba-radix-cache-strategy extra_buffer_lazy --mamba-max-states-per-path 2 --sleep-on-idle"
    if [ "${DRY_RUN:-0}" != 1 ]; then
      [ -f "$ROOT/$MOE_SNAPSHOT/model-00003-of-00003.safetensors" ] || { echo "error: weights missing under $ROOT/$MOE_SNAPSHOT"; exit 1; }
    fi
    ;;
  *) echo "error: unknown PROFILE=$PROFILE (27b-mtp | 27b | moe)"; exit 1 ;;
esac
MAX_RUNNING=${MAX_RUNNING:-$PROFILE_MAX_RUNNING}
IMAGE=${IMAGE:-${PROFILE_IMAGE:-$IMAGE_DEFAULT}}

mkdir -p "$ROOT/huggingface" "$ROOT/sglang-cache" "$ROOT/templates"
if [ ! -s "$ROOT/api_key" ]; then
  umask 077
  openssl rand -hex 24 > "$ROOT/api_key"
  openssl rand -hex 24 > "$ROOT/admin_api_key"
fi
API_KEY=$(cat "$ROOT/api_key")
ADMIN_KEY=$(cat "$ROOT/admin_api_key")

if [ "${DRY_RUN:-0}" != 1 ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # Wait for the old server's VRAM to drain before measuring the desktop's share (an inflated reading shrinks the pool).
  for _ in $(seq 1 30); do
    USED_NOW=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ,')
    [ "$USED_NOW" -lt 12000 ] && break
    sleep 1
  done
  sleep 2
fi

# --- size the static memory fraction from what the Windows desktop is using right now ---
read -r USED_MIB TOTAL_MIB < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
if [ "$USED_MIB" -lt "${DESKTOP_MIN_MIB:-0}" ]; then USED_MIB=${DESKTOP_MIN_MIB}; fi
HEADROOM_MIB=${HEADROOM_MIB:-$PROFILE_HEADROOM}
NONSTATIC_MIB=${NONSTATIC_MIB:-$PROFILE_NONSTATIC}
WDDM_RESERVE_MIB=420                  # nvidia-smi memory.reserved under WSL2: a constant 420 MiB on this card
CUDA_HIDDEN_MIB=1826                  # CUDA under WSL2 reports total minus ~1.8 GiB as available
AUTO_FRACTION=$(awk -v t="$TOTAL_MIB" -v u="$USED_MIB" -v h="$HEADROOM_MIB" -v n="$NONSTATIC_MIB" -v w="$WDDM_RESERVE_MIB" -v c="$CUDA_HIDDEN_MIB" \
  'BEGIN { f = (t - w - u - h - n) / (t - c); if (f > 0.92) f = 0.92; if (f < 0.70) f = 0.70; printf "%.3f", int(f * 1000) / 1000 }')
MEM_FRACTION=${MEM_FRACTION:-$AUTO_FRACTION}
echo "desktop VRAM in use: ${USED_MIB} MiB of ${TOTAL_MIB} MiB -> --mem-fraction-static ${MEM_FRACTION} (headroom ${HEADROOM_MIB} MiB)"

# --- the KV pool this launch will get, and the window ---
# SGLang v0.5.20 sizing: pool tokens = (fraction x CUDA-visible MiB - weights - 100 MiB VLM cache - SLOT_MIB x (slots + 1)
# - 12) x 1024 / KiB per token. Estimate lands within 0.4% of the allocated pool on this card.
KV_DTYPE=${KV_DTYPE:-fp8_e4m3}
case "$KV_DTYPE" in
  # FP4 cell = FP8/2 + FP8/16 (block scales) + one shared FP8 dequant row of kv_heads x 256 x 2 B per token.
  nvfp4|fp4_mx_block16) KV_KIB=$(awk -v k="$KV_KIB_FP8" -v h="$KV_HEADS" 'BEGIN { printf "%.4f", k * 0.5625 + h * 256 * 2 / 1024 }') ;;
  bfloat16|bf16)        KV_KIB=$((KV_KIB_FP8 * 2)) ;;
  *)                    KV_KIB=$KV_KIB_FP8 ;;
esac
EST_POOL=$(awk -v f="$MEM_FRACTION" -v p="$((TOTAL_MIB - CUDA_HIDDEN_MIB))" -v w="$WEIGHT_MIB" -v s="$MAMBA_SLOTS" -v m="$SLOT_MIB" -v k="$KV_KIB" \
  'BEGIN { printf "%d", (f * p - w - 100 - m * (s + 1) - 12) * 1024 / k }')
# The window must fit the pool: SGLang would otherwise cap request length silently and abort long prompts.
if [ $((EST_POOL - 2048)) -ge "$WANT_CONTEXT" ]; then
  CONTEXT_LENGTH=$WANT_CONTEXT
else
  CONTEXT_LENGTH=$(( (EST_POOL - 2048) / 1024 * 1024 ))
  echo "WARNING: profile $PROFILE wants a ${WANT_CONTEXT}-token window but this launch's pool is only ~${EST_POOL}; using ${CONTEXT_LENGTH}. Close GPU-heavy apps and relaunch for the full window"
fi
echo "profile=${PROFILE} slots=${MAMBA_SLOTS} estimated KV pool (${KV_DTYPE}): ${EST_POOL} tokens -> --context-length ${CONTEXT_LENGTH}"

# shellcheck disable=SC2206  # PROFILE_ARGS / EXTRA_ARGS are intentionally word-split
# --model-type llm skips the diffusion auto-detect (~2.4-5 s per boot).
SERVE_ARGS=(sglang serve --model-type llm
    --model-path "$MODEL"
    --served-model-name "$SERVED_NAME"
    --context-length "$CONTEXT_LENGTH"
    --kv-cache-dtype "$KV_DTYPE"
    --attention-backend flashinfer
    --mamba-ssm-dtype bfloat16
    --mem-fraction-static "$MEM_FRACTION"
    --max-mamba-cache-size "$MAMBA_SLOTS"
    --disable-prefill-cuda-graph
    --chunked-prefill-size 2048
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$MAX_RUNNING"
    --cuda-graph-bs-decode $(seq 1 "$MAX_RUNNING")
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder
    --api-key "$API_KEY" --admin-api-key "$ADMIN_KEY"
    --enable-metrics --enable-cache-report
    --host 0.0.0.0 --port 30000 $PROFILE_ARGS ${EXTRA_ARGS:-})

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%s\n' "${SERVE_ARGS[@]}" | sed "s/^$API_KEY\$/<API_KEY>/; s/^$ADMIN_KEY\$/<ADMIN_KEY>/"
  exit 0
fi

docker run -d --name "$NAME" --gpus all --ipc=host \
  -p "127.0.0.1:${HOST_PORT:-30000}:30000" \
  -v "$ROOT/huggingface:/root/.cache/huggingface" \
  -v "$ROOT/sglang-cache:/root/.cache/sglang" \
  -v "$ROOT/templates:/templates:ro" \
  -e SGLANG_CACHE_DIR=/root/.cache/sglang \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPYCACHEPREFIX="/root/.cache/sglang/pycache/${IMAGE##*:}" \
  -e SGLANG_EMPTY_CACHE_INTERVAL="${EMPTY_CACHE_INTERVAL:-30}" \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${HEALTH_GENERATE:-false}" \
  -e SGLANG_FLASHINFER_AUTOTUNE_EXTEND="${AUTOTUNE_EXTEND:-1}" \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-$PROFILE_SKIP_LOCK}" \
  -e SGLANG_OFFLOAD_EMBEDDING_TO_HOST="${OFFLOAD_EMBEDDING:-$PROFILE_OFFLOAD_EMBEDDING}" \
  "$IMAGE" "${SERVE_ARGS[@]}"
# Container env (all present in v0.5.20 environ.py, except OFFLOAD_EMBEDDING, which the embhost image adds):
#   PYTHONPYCACHEPREFIX  bytecode survives container recreation (~13 s per boot).
#   SGLANG_EMPTY_CACHE_INTERVAL  with --sleep-on-idle, return PyTorch's cached blocks to the driver every N s of idle
#     time. The default (-1) keeps every allocation spike forever, which is what pushed the card into WDDM paging.
#   SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false  /health stops running a 1-token generation every ~30 s.
#   SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1  also tunes GEMM tactics for prefill-sized batches (cached per model path).

echo "started container $NAME (profile $PROFILE; logs: docker logs -f $NAME)"
