#!/usr/bin/env bash
# Start, stop or inspect the SGLang server on a rented RTX 5090, natively inside the instance (no Docker needed).
#   bash deploy/vastai/serve.sh                 start PROFILE=27b-mtp in the background and wait until it answers
#   PROFILE=moe bash deploy/vastai/serve.sh     the 35B-A3B MoE instead (one model per GPU; stop the other first)
#   bash deploy/vastai/serve.sh stop | restart | status | logs
#   DRY_RUN=1 bash deploy/vastai/serve.sh       print the environment and arguments (keys masked), start nothing
# The server listens on 127.0.0.1:$PORT (default 30000) only. Reach it through an SSH tunnel (README.md next to
# this file); clients need the key in $SERVE_ROOT/api_key. Never publish this port: SGLang is not built to face the
# internet.
#
# Profiles are the configurations we ran and measured on our own RTX 5090:
#   27b-mtp  Qwen3.8-27B NVFP4 (gittensor) + grafted MTP head: EAGLE 3 steps / top-k 1 / 4 draft tokens with
#            ReplaySSM, input embedding in pinned host RAM, k1's four output-identical switches, decode lock skipped
#            -> 7 requests at once, 147,456-token window. HICACHE_GB=<GB>|max adds the host-RAM KV tier (k2).
#   27b      the same weights without MTP (the rollback), 5 at once
#   27b-dflash  the same target with a DFlash2 block-diffusion drafter instead of the MTP head (8-token blocks,
#            2,048-token draft window): DFLASH_DRAFT=nvfp4 (default) | bf16. 3 at once by default: without ReplaySSM
#            (KDA-only with DFLASH in v0.5.20) verify keeps a bf16 GDN state per draft token per running request,
#            ~73 MiB x 8 x (MAX_RUNNING + 1). See docs/DFLASH.md.
#   moe      Qwen3.6-35B-A3B NVFP4 with the flashinfer_cutlass MoE runner, 8 at once, 131,072-token window
# Knobs (env): MAX_RUNNING (state slots follow), MAMBA_SLOTS, CONTEXT_LENGTH, MEM_FRACTION, HEADROOM_MIB, HICACHE_GB,
# HICACHE_RESERVE_GB, KV_DTYPE, PORT, K1=0 (k1 switches off), DFLASH_DRAFT, DFLASH_BLOCK, DFLASH_WINDOW, EXTRA_ARGS,
# BOOT_TIMEOUT (s).
set -euo pipefail
# shellcheck source=deploy/vastai/env.sh
source "$(dirname "$0")/env.sh"
CMD=${1:-start}
PROFILE=${PROFILE:-27b-mtp}
PORT=${PORT:-30000}
PIDF=$SERVE_ROOT/server.pid
LOG=$SERVE_ROOT/logs/server.log
KEYF=$SERVE_ROOT/api_key
die() { echo "error: $*" >&2; exit 1; }
no_keys() { grep -v 'server_args=' || true; }   # SGLang's startup line prints both API keys
running() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
health() {
  curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $(cat "$KEYF")" \
    "http://127.0.0.1:$PORT/health" 2>/dev/null || true
}

do_stop() {
  if running; then
    local pid
    pid=$(cat "$PIDF")
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    echo "stopped the server (pid $pid)"
  else
    echo "server not running"
  fi
  rm -f "$PIDF"
}

do_status() {
  if running; then echo "server pid $(cat "$PIDF"), /health -> $(health)"; else echo "server not running"; fi
  if [ -f "$LOG" ]; then grep -hE 'max_total_num_tokens=|hierarchical KV host pool|fired up' "$LOG" | no_keys | tail -3 || true; fi
  nvidia-smi --query-gpu=memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null || true
}

case "$CMD" in
  stop) do_stop; exit 0 ;;
  status) do_status; exit 0 ;;
  logs) if [ -f "$LOG" ]; then tail -n "${LINES:-60}" "$LOG" | no_keys || true; fi; exit 0 ;;
  restart) do_stop ;;
  start) ;;
  *) die "usage: serve.sh [start|stop|restart|status|logs]" ;;
esac

if running; then echo "already running (pid $(cat "$PIDF")); stop it first: bash deploy/vastai/serve.sh stop"; exit 0; fi
[ -s "$KEYF" ] || die "no API key in $SERVE_ROOT: run deploy/vastai/bootstrap.sh first"
[ -f "$SGL_SRC/.rtx5090-patches" ] || die "SGLang is not patched yet: run deploy/vastai/bootstrap.sh first"

# --- profile -----------------------------------------------------------------------------------------------------
SKIP_LOCK=0; OFFLOAD=0; NONSTATIC=2600; SCRATCH_MIB=0; ENV_EXTRA=()
TPL=$SERVE_ROOT/templates/qwen3.8-upstream.jinja
case "$PROFILE" in
  27b-mtp|27b|27b-dflash)
    MODEL=$(snapshot_dir "$GT_REPO" "$GT_REV")
    [ -f "$MODEL/model-00002-of-00002.safetensors" ] || die "27B weights missing: MODELS=27b-mtp bash deploy/vastai/bootstrap.sh"
    SERVED=qwen3.8-27b
    WEIGHT_MIB=17510; KV_KIB=32; KV_HEADS=4; SLOT_MIB=74.8125   # 16 full-attention layers x 4 KV heads x 512 B
    SLOTS_DEF=22; RUN_DEF=5; WANT_DEF=147456                     # 131,072 input + 16,384 output
    PROFILE_ARGS=(--mamba-radix-cache-strategy extra_buffer_lazy --mamba-max-states-per-path 2 --sleep-on-idle)
    if [ -s "$TPL" ]; then PROFILE_ARGS=(--chat-template "$TPL" "${PROFILE_ARGS[@]}"); fi
    if [ "$PROFILE" = 27b-mtp ]; then
      MODEL=$HF_HOME/local/gittensor-mtp
      [ -s "$MODEL/mtp.safetensors" ] || die "MTP graft missing: bash deploy/vastai/bootstrap.sh"
      WEIGHT_MIB=15700        # target without the host-side embedding + bf16 MTP layer + ReplaySSM ring
      KV_KIB=34               # EAGLE adds the MTP layer's own KV
      NONSTATIC=3000          # draft/verify CUDA graphs and buffers
      RUN_DEF=7; SKIP_LOCK=1; OFFLOAD=1
      PROFILE_ARGS+=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
                     --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec)
      if [ "${K1:-1}" = 1 ]; then
        # the four k1 switches that keep temperature-0 output identical (53/53 test answers); see docs/K1_KERNELS.md
        ENV_EXTRA+=(SGLANG_KV_SKIP_UNIT_SCALE_DIV=1 SGLANG_AUTOTUNE_TARGET_LMHEAD=1
                    SGLANG_ATTN_MLP_SILU_FP4_FUSION=1 SGLANG_EAGLE_SYNCFREE_SEAM=1)
      fi
    fi
    if [ "$PROFILE" = 27b-dflash ]; then
      case "${DFLASH_DRAFT:-nvfp4}" in
        nvfp4) DRAFT=$(snapshot_dir "$DF_NVFP4_REPO" "$DF_NVFP4_REV"); DRAFT_MIB=1638 ;;   # boot log: 'mem usage=1.60 GB'
        bf16)  DRAFT=$(snapshot_dir "$DF_BF16_REPO" "$DF_BF16_REV"); DRAFT_MIB=3670 ;;     # size of the bf16 file
        *) die "unknown DFLASH_DRAFT=${DFLASH_DRAFT} (nvfp4 | bf16)" ;;
      esac
      [ -s "$DRAFT/model.safetensors" ] || die "DFlash2 drafter missing: MODELS=27b-dflash bash deploy/vastai/bootstrap.sh"
      BLOCK=${DFLASH_BLOCK:-8}
      WEIGHT_MIB=$((15084 + DRAFT_MIB))   # target with the embedding in host RAM (boot log: 14.73 GB) + the drafter
      KV_KIB=42                           # target 32 + draft 5 layers x 8 KV heads x 256 x 1 B (one slot per target token)
      NONSTATIC=3000
      RUN_DEF=3; SKIP_LOCK=1; OFFLOAD=1
      # filled in once MAX_RUNNING is known: ~73 MiB per (running request + 1) per draft token
      SCRATCH_PER_MIB=$((73 * BLOCK))
      PROFILE_ARGS+=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT"
                     --speculative-dflash-block-size "$BLOCK" --speculative-draft-window-size "${DFLASH_WINDOW:-2048}")
      if [ "${K1:-1}" = 1 ]; then
        # the k1 switches minus the EAGLE seam, which does not apply to DFLASH
        ENV_EXTRA+=(SGLANG_KV_SKIP_UNIT_SCALE_DIV=1 SGLANG_AUTOTUNE_TARGET_LMHEAD=1 SGLANG_ATTN_MLP_SILU_FP4_FUSION=1)
      fi
    fi
    ;;
  moe)
    MODEL=$(snapshot_dir "$MOE_REPO" "$MOE_REV")
    [ -f "$MODEL/model-00003-of-00003.safetensors" ] || die "MoE weights missing: MODELS=moe bash deploy/vastai/bootstrap.sh"
    SERVED=qwen3.6-35b-a3b
    WEIGHT_MIB=21170; KV_KIB=10; KV_HEADS=2; SLOT_MIB=31.3      # 10 full-attention layers x 2 KV heads x 512 B
    SLOTS_DEF=32; RUN_DEF=8; WANT_DEF=131072; SKIP_LOCK=1
    # v0.5.20 picks the SM100-only trtllm MoE runner for NVFP4, which dies on sm_120; flashinfer_cutlass works.
    PROFILE_ARGS=(--moe-runner-backend flashinfer_cutlass --mamba-radix-cache-strategy extra_buffer_lazy
                  --mamba-max-states-per-path 2 --sleep-on-idle)
    ;;
  *) die "unknown PROFILE=$PROFILE (27b-mtp | 27b | 27b-dflash | moe)" ;;
esac

# --- sizing ------------------------------------------------------------------------------------------------------
# A running request pins 3 state slots with the decode lock skipped, 4 without (extra_buffer_lazy); v0.5.20 caps
# running requests at slots / that ratio, so the slot count follows MAX_RUNNING.
MAX_RUNNING=${MAX_RUNNING:-$RUN_DEF}
RATIO=4; [ "$SKIP_LOCK" = 1 ] && RATIO=3
MIN_SLOTS=$((RATIO * MAX_RUNNING + 1))
MAMBA_SLOTS=${MAMBA_SLOTS:-$((SLOTS_DEF > MIN_SLOTS ? SLOTS_DEF : MIN_SLOTS))}
if [ "$PROFILE" = 27b-dflash ]; then SCRATCH_MIB=$((SCRATCH_PER_MIB * (MAX_RUNNING + 1))); fi
WANT=${CONTEXT_LENGTH:-$WANT_DEF}
HEADROOM=${HEADROOM_MIB:-1024}   # allocator spikes; nothing else shares the GPU on a rented box
if [ -n "${GPU_TOTAL_MIB:-}" ]; then          # dry runs on a machine without the GPU
  TOTAL=$GPU_TOTAL_MIB; FREE=${GPU_FREE_MIB:-$GPU_TOTAL_MIB}
else
  read -r FREE TOTAL < <(python3 -c 'import torch; f, t = torch.cuda.mem_get_info(0); print(f >> 20, t >> 20)') || true
fi
[ -n "${TOTAL:-}" ] || die "could not read GPU memory through torch (is the GPU visible? nvidia-smi)"
AUTO_F=$(awk -v t="$TOTAL" -v f="$FREE" -v n="$NONSTATIC" -v h="$HEADROOM" \
  'BEGIN { x = (f - n - h) / t; if (x > 0.92) x = 0.92; if (x < 0.60) x = 0.60; printf "%.3f", int(x * 1000) / 1000 }')
MEM_FRACTION=${MEM_FRACTION:-$AUTO_F}
KV_DTYPE=${KV_DTYPE:-fp8_e4m3}
case "$KV_DTYPE" in
  nvfp4|fp4_mx_block16) KV_EFF=$(awk -v k="$KV_KIB" -v h="$KV_HEADS" 'BEGIN { printf "%.4f", k * 0.5625 + h * 256 * 2 / 1024 }') ;;
  bfloat16|bf16)        KV_EFF=$((KV_KIB * 2)) ;;
  *)                    KV_EFF=$KV_KIB ;;
esac
# SGLang v0.5.20 pool = (fraction x CUDA total MiB - weights - 100 - slot MiB x (slots + 1) - 12 - verify scratch)
# x 1024 / KiB per token (scratch is the DFLASH per-draft-token GDN state; 0 otherwise; estimate ~2% high there)
EST_POOL=$(awk -v f="$MEM_FRACTION" -v p="$TOTAL" -v w="$WEIGHT_MIB" -v s="$MAMBA_SLOTS" -v m="$SLOT_MIB" -v k="$KV_EFF" -v x="$SCRATCH_MIB" \
  'BEGIN { printf "%d", (f * p - w - 100 - m * (s + 1) - 12 - x) * 1024 / k }')
if [ $((EST_POOL - 2048)) -ge "$WANT" ]; then
  CTX=$WANT
else
  CTX=$(((EST_POOL - 2048) / 1024 * 1024))
  echo "WARNING: wanted a ${WANT}-token window but the KV pool is only ~${EST_POOL} tokens; using ${CTX}"
fi
[ "$CTX" -ge 8192 ] || die "the KV pool would be only ~${EST_POOL} tokens: lower MAX_RUNNING / MAMBA_SLOTS or free GPU memory"

# --- optional host-RAM KV tier (HiCache, k2) -------------------------------------------------------------------
HC=${HICACHE_GB:-0}
HICACHE_ARGS=(); HC_TXT=""
case "$HC" in max|0) ;; ''|*[!0-9]*) die "HICACHE_GB must be a number of GB or 'max'" ;; esac
if [ "$HC" != 0 ]; then
  # 27b-dflash: the host pools (incl. the draft's sidecar KV pool) allocate and serve; a restore test is still to do
  case "$PROFILE" in 27b-mtp|27b-dflash) ;; *) die "HICACHE_GB is only validated with PROFILE=27b-mtp (and allocates with 27b-dflash)" ;; esac
  RESERVE=${HICACHE_RESERVE_GB:-6}
  # RAM this container may still use: /proc/meminfo shows the host, the cgroup shows the rented share
  AVAIL_GIB=$(python3 - <<'PY'
def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None
avail = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")}["MemAvailable"]
for lim_f, cur_f, stat_f, key in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "inactive_file"),
                                 ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                                  "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")):
    lim, cur = rd(lim_f), rd(cur_f)
    if lim and cur and lim != "max" and int(lim) < 1 << 60:
        st = dict(l.split() for l in (rd(stat_f) or "").splitlines() if len(l.split()) == 2)
        avail = min(avail, int(lim) - (int(cur) - int(st.get(key, 0))))
        break
print(f"{avail / 2**30:.1f}")
PY
)
  # the server itself takes ~9 GiB of RAM before SGLang sizes the pools; SGLang then keeps RESERVE free; 2 GiB margin
  FIT=$(awk -v a="$AVAIL_GIB" -v r="$RESERVE" 'BEGIN { g = int((a - 9 - r - 2) * 1.0737); if (g < 0) g = 0; print g }')
  if [ "$HC" = max ]; then HC=$FIT
  elif [ "$HC" -gt "$FIT" ]; then echo "WARNING: a ${HC} GB host cache does not fit (${AVAIL_GIB} GiB RAM available); using ${FIT} GB"; HC=$FIT
  fi
  if [ "$HC" -lt 4 ]; then echo "note: only ${FIT} GB of RAM fits a host cache here (${AVAIL_GIB} GiB available); serving without it"; HC=0; fi
fi
if [ "$HC" != 0 ]; then
  ENV_EXTRA+=(SGLANG_HICACHE_HOST_ALLOC=cuda SGLANG_HICACHE_NO_STAGED_WRITE_BACK=1
              SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN=1 SGLANG_HICACHE_HOST_RESERVE_GB="$RESERVE")
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-size "$HC" --hicache-mem-layout page_first
                --hicache-io-backend kernel --hicache-write-policy write_through)
  HC_TXT=", host-RAM cache ${HC} GB"
fi

# --- command line ------------------------------------------------------------------------------------------------
API_KEY=$(cat "$KEYF"); ADMIN_KEY=$(cat "$SERVE_ROOT/admin_api_key")
ENV_ALL=(SGLANG_CACHE_DIR="$SERVE_ROOT/cache" HF_HUB_OFFLINE=1
         SGLANG_EMPTY_CACHE_INTERVAL="${EMPTY_CACHE_INTERVAL:-30}"   # return cached blocks after 30 s idle
         SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false              # /health without a 1-token generation
         SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1                         # also tune GEMM tactics for prefill sizes
         SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="$SKIP_LOCK"
         SGLANG_OFFLOAD_EMBEDDING_TO_HOST="$OFFLOAD"
         "${ENV_EXTRA[@]}")
ARGS=(sglang serve --model-type llm
      --model-path "$MODEL" --served-model-name "$SERVED"
      --context-length "$CTX" --kv-cache-dtype "$KV_DTYPE"
      --attention-backend flashinfer --mamba-ssm-dtype bfloat16
      --mem-fraction-static "$MEM_FRACTION" --max-mamba-cache-size "$MAMBA_SLOTS"
      --disable-prefill-cuda-graph --chunked-prefill-size 2048
      --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs-decode "$MAX_RUNNING"
      --cuda-graph-bs-decode $(seq 1 "$MAX_RUNNING")
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder
      --api-key "$API_KEY" --admin-api-key "$ADMIN_KEY"
      --enable-metrics --enable-cache-report
      --host 127.0.0.1 --port "$PORT"
      "${PROFILE_ARGS[@]}" "${HICACHE_ARGS[@]}")
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
if [ -n "${EXTRA_ARGS:-}" ]; then ARGS+=($EXTRA_ARGS); fi

echo "profile $PROFILE ($SERVED): $MAX_RUNNING at once, $MAMBA_SLOTS state slots, --mem-fraction-static $MEM_FRACTION (GPU $FREE of $TOTAL MiB free)"
echo "estimated KV pool ~${EST_POOL} tokens ($KV_DTYPE) -> window ${CTX} tokens${HC_TXT}"
if [ "${DRY_RUN:-0}" = 1 ]; then
  printf 'env %s\n' "${ENV_ALL[@]}"
  printf '%s\n' "${ARGS[@]}" | sed "s/^$API_KEY\$/<API_KEY>/; s/^$ADMIN_KEY\$/<ADMIN_KEY>/"
  exit 0
fi

# --- launch in its own session (stop = kill the whole process group) and wait for /health --------------------
: > "$LOG"; chmod 600 "$LOG"
cd "$SERVE_ROOT"
# SGLang copies every SGL_* variable to SGLANG_* (a deprecated prefix), so do not hand it test-script variables
for v in $(compgen -e | grep '^SGL_' || true); do unset "$v"; done
env "${ENV_ALL[@]}" setsid nohup "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDF"
T0=$(date +%s); LAST=0
echo "loading (pid $PID). The first start on a new box compiles and autotunes kernels: allow 2-5 minutes."
while :; do
  [ "$(health)" = 200 ] && break
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: the server exited during start-up; last log lines:" >&2
    tail -n 40 "$LOG" | no_keys >&2
    rm -f "$PIDF"; exit 1
  fi
  NOW=$(($(date +%s) - T0))
  [ "$NOW" -lt "${BOOT_TIMEOUT:-1200}" ] || die "not healthy after ${NOW} s; see: bash deploy/vastai/serve.sh logs"
  if [ $((NOW - LAST)) -ge 15 ]; then echo "  ... ${NOW} s"; LAST=$NOW; fi
  sleep 3
done
echo "ready in $(($(date +%s) - T0)) s"
grep -hE 'max_total_num_tokens=|hierarchical KV host pool' "$LOG" | no_keys | sed 's/^/  /' || true
echo "API  http://127.0.0.1:$PORT/v1   model $SERVED   key: cat $KEYF"
echo "test bash deploy/vastai/check.sh"
