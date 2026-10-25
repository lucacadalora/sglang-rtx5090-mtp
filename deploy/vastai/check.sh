#!/usr/bin/env bash
# Acceptance test after serve.sh: the functional gates we ran before every change at home (auth, thinking on/off,
# tool calls with typed arguments, image input, short decode) and a speed probe at 1, 4 and full concurrency.
#   bash deploy/vastai/check.sh
#   STREAMS=1,8,16 bash deploy/vastai/check.sh     # other concurrency levels for the speed probe
set -euo pipefail
# shellcheck source=deploy/vastai/env.sh
source "$(dirname "$0")/env.sh"
PORT=${PORT:-30000}
BASE=http://127.0.0.1:$PORT
KEY=$(cat "$SERVE_ROOT/api_key")
MODEL=$(curl -sf -m 10 -H "Authorization: Bearer $KEY" "$BASE/v1/models" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])') \
  || { echo "error: no server answering on $BASE (start it: bash deploy/vastai/serve.sh)" >&2; exit 1; }
echo "== gates: $MODEL"
SGL_ROOT=$SERVE_ROOT SGL_BASE=$BASE SGL_MODEL=$MODEL python3 "$REPO/bench/gates.py" quick
echo
echo "== speed"
python3 "$(dirname "$0")/speed.py" --base "$BASE" --model "$MODEL" --key-file "$SERVE_ROOT/api_key" \
  --log "$SERVE_ROOT/logs/server.log" ${STREAMS:+--streams "$STREAMS"}
