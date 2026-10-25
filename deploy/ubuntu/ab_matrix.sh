#!/usr/bin/env bash
# A/B matrix for Qwen3.8-27B on one RTX 5090: the MTP head vs the DFlash2 drafter (NVFP4 or bf16), at several
# concurrency levels and on three workloads. Runs where deploy/vastai runs (inside the container on your own box, or on
# a rented instance), after:  MODELS="27b-mtp 27b-dflash" bash deploy/vastai/bootstrap.sh
#   bash deploy/ubuntu/ab_matrix.sh                                  # the full matrix, ~45-60 min
#   ARMS="mtp:7 dflash-bf16:1" WORKLOADS=code bash deploy/ubuntu/ab_matrix.sh   # a subset
# ARMS: <arm>[:max_running], arm = mtp | dflash-nvfp4 | dflash-bf16. Each arm: serve.sh restart -> one warm-up reply ->
# quick gates (GATES=0 skips) -> speed.py per workload at 1, 3 and 7 streams (those <= max_running). TEMP (default 0)
# is the sampling temperature; EXTRA_ENV (e.g. "HICACHE_GB=max") is passed to every serve.sh start.
# Output: $SERVE_ROOT/ab/matrix-<stamp>.jsonl (one line per arm x workload x stream level) and .txt (the full log,
# with each arm's boot facts: KV pool, window, verify scratch, free VRAM). The last step restarts PROFILE=27b-mtp
# (RESTORE=0 leaves the last arm running).
set -euo pipefail
# shellcheck source=deploy/vastai/env.sh
source "$(dirname "$0")/../vastai/env.sh"
V=$REPO/deploy/vastai
ARMS=${ARMS:-"mtp:7 dflash-nvfp4:1 dflash-nvfp4:3 dflash-nvfp4:7 dflash-bf16:1 dflash-bf16:3 dflash-bf16:7"}
WORKLOADS=${WORKLOADS:-"prose code structured"}
TEMP=${TEMP:-0}
PORT=${PORT:-30000}
LOG=$SERVE_ROOT/logs/server.log
mkdir -p "$SERVE_ROOT/ab"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$SERVE_ROOT/ab/matrix-$STAMP.jsonl
SUM=$SERVE_ROOT/ab/matrix-$STAMP.txt
say() { echo "$*" | tee -a "$SUM"; }
say "matrix $STAMP: arms [$ARMS], workloads [$WORKLOADS], temperature $TEMP"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader | tee -a "$SUM"

for arm in $ARMS; do
  name=${arm%%:*}; run=""
  [ "$arm" != "$name" ] && run=${arm#*:}
  case "$name" in
    mtp)          envs=(PROFILE=27b-mtp) ;;
    dflash-nvfp4) envs=(PROFILE=27b-dflash DFLASH_DRAFT=nvfp4) ;;
    dflash-bf16)  envs=(PROFILE=27b-dflash DFLASH_DRAFT=bf16) ;;
    *) echo "error: unknown arm '$name' (mtp | dflash-nvfp4 | dflash-bf16)" >&2; exit 1 ;;
  esac
  if [ -n "$run" ]; then envs+=(MAX_RUNNING="$run"); fi
  # shellcheck disable=SC2206  # EXTRA_ENV is intentionally word-split into NAME=VALUE pairs
  if [ -n "${EXTRA_ENV:-}" ]; then envs+=($EXTRA_ENV); fi
  label=$name${run:+-r$run}
  say ""; say "== $label (${envs[*]})"
  bash "$V/serve.sh" stop >/dev/null 2>&1 || true
  if ! env "${envs[@]}" bash "$V/serve.sh" start 2>&1 | tee -a "$SUM"; then
    say "   $label did not start; skipped (see the log lines above)"
    continue
  fi
  say "   boot: $(grep -ohE 'max_total_num_tokens=[0-9]+|context_len=[0-9]+|max_running_requests=[0-9]+|available_gpu_mem=[0-9.]+ GB|intermediate_ssm_state_cache size: [0-9.]+GB' "$LOG" | sort -u | tr '\n' ' ')"
  maxrun=$(grep -ohE 'max_running_requests=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
  levels=""
  for n in 1 3 7; do [ "$n" -le "${maxrun:-1}" ] && levels=${levels:+$levels,}$n; done
  # warm-up: the first replies after a start autotune kernels; keep them out of the numbers
  python3 "$V/speed.py" --model qwen3.8-27b --key-file "$SERVE_ROOT/api_key" --streams 1 --max-tokens 64 >/dev/null 2>&1 || true
  if [ "${GATES:-1}" = 1 ]; then
    # gates.py prints "QUICK: PASS|FAIL" (its exit code does not carry the verdict)
    SGL_ROOT=$SERVE_ROOT SGL_BASE=http://127.0.0.1:$PORT SGL_MODEL=qwen3.8-27b \
      SGL_TOKENIZER="$(snapshot_dir "$GT_REPO" "$GT_REV")/tokenizer.json" \
      python3 "$REPO/bench/gates.py" quick > "$SUM.gates" 2>&1 || true
    if grep -q '^QUICK: PASS' "$SUM.gates"; then
      say "   gates: PASS"
    else
      say "   gates: FAIL"; tail -n 20 "$SUM.gates" | tee -a "$SUM"
    fi
  fi
  for w in $WORKLOADS; do
    python3 "$V/speed.py" --model qwen3.8-27b --key-file "$SERVE_ROOT/api_key" --log "$LOG" --streams "$levels" \
      --workload "$w" --temperature "$TEMP" --json-out "$OUT" --label "$label" 2>&1 | tee -a "$SUM"
  done
done

say ""; say "== summary (server-side gen throughput, tok/s; accept = mean accepted tokens per verify step)"
python3 - "$OUT" <<'PY' | tee -a "$SUM"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if sys.argv[1] else []
cells = {}
for r in rows:
    cells.setdefault((r["label"], r["workload"]), {})[r["streams"]] = r
work = sorted({w for _, w in cells}, key=["prose", "code", "structured", "mixed"].index)
print(f"{'arm':<18} {'workload':<11} {'1 stream':>16} {'3 streams':>16} {'7 streams':>16}")
for (label, w), by in sorted(cells.items(), key=lambda kv: (kv[0][0], work.index(kv[0][1]))):
    def cell(n):
        r = by.get(n)
        if not r or not r.get("server"):
            return "-"
        return f"{r['server']:.0f}" + (f" ({r['accept_len']:.2f})" if r.get("accept_len") else "")
    print(f"{label:<18} {w:<11} {cell(1):>16} {cell(3):>16} {cell(7):>16}")
PY
say "rows: $OUT"

if [ "${RESTORE:-1}" = 1 ]; then
  say ""; say "== restoring PROFILE=27b-mtp"
  bash "$V/serve.sh" stop >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  env PROFILE=27b-mtp ${EXTRA_ENV:-} bash "$V/serve.sh" start 2>&1 | tail -n 4 | tee -a "$SUM" || true
fi
