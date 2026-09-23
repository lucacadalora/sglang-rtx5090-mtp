#!/usr/bin/env bash
# Build both images (CPU only), then run the WSL2 go/no-go test for UVA reads of a pinned host table in a throwaway
# GPU container: bit-exact against a device lookup, CUDA-graph capture and replay, and gather timing.
# Run from anywhere inside WSL (as a user that can run docker):  bash docker/build.sh
set -euo pipefail
cd "$(dirname "$0")"
for f in mtpfix/pr37155.diff embhost/*.py embhost/Dockerfile mtpfix/Dockerfile; do
  if grep -q $'\r' "$f"; then echo "error: $f has CRLF line endings (clone with core.autocrlf=false)"; exit 1; fi
done
docker build -t local/sglang:v0.5.20-cu130-mtpfix mtpfix
docker build -t local/sglang:v0.5.20-cu130-mtpfix-embhost embhost
docker run --rm --gpus all --entrypoint python3 local/sglang:v0.5.20-cu130-mtpfix-embhost \
  /opt/embed_offload/test_host_gather_wsl.py
