#!/usr/bin/env bash
# Run the stock SGLang image as a long-lived container on your own Ubuntu box with one RTX 5090, then use the
# deploy/vastai scripts inside it (they patch SGLang in place and serve natively, exactly as on a rented card).
#   bash deploy/ubuntu/container.sh up       # pull the image, start the container, clone/update this repo inside
#   bash deploy/ubuntu/container.sh shell    # root shell inside the container, in the repo
#   bash deploy/ubuntu/container.sh run CMD  # run one command inside, in the repo (e.g. bash deploy/ubuntu/ab_matrix.sh)
#   bash deploy/ubuntu/container.sh down     # stop and remove the container (weights and keys stay in $WORK)
# WORK (default $HOME/sglang-work) is mounted at /workspace: weights (hf/), keys, template, logs (sglang/) and the repo
# clone survive the container. --network host: the server's 127.0.0.1:30000 is the host's loopback, so an SSH tunnel
# to the host reaches it and nothing is published to the network.
set -euo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.20-cu130}
NAME=${CONTAINER_NAME:-sglang-5090}
WORK=${WORK:-$HOME/sglang-work}
REPO_URL=${REPO_URL:-https://github.com/lucacadalora/sglang-rtx5090-mtp}
IN_REPO=/workspace/sglang-rtx5090-mtp
die() { echo "error: $*" >&2; exit 1; }

case "${1:-up}" in
  up)
    command -v docker >/dev/null || die "docker is not installed (deploy/ubuntu/README.md, step 1)"
    docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (is your user in the docker group? log out and in again)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
    mkdir -p "$WORK"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      docker start "$NAME" >/dev/null
      echo "started the existing container $NAME"
    else
      docker pull "$IMAGE"
      # --ipc=host + large shm for PyTorch/NCCL; memlock unlimited for the pinned host-RAM tables (HiCache, embedding)
      docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 16g --ulimit memlock=-1 --network host \
        -v "$WORK:/workspace" --entrypoint sleep "$IMAGE" infinity >/dev/null
      echo "created container $NAME from $IMAGE (workspace $WORK)"
    fi
    docker exec "$NAME" nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader \
      || die "the container cannot see the GPU: configure the NVIDIA Container Toolkit for Docker (README step 1)"
    docker exec "$NAME" bash -c "if [ -d $IN_REPO/.git ]; then git -C $IN_REPO pull --ff-only; \
      else git -c core.autocrlf=false clone $REPO_URL $IN_REPO; fi"
    echo "next: bash deploy/ubuntu/container.sh shell   (then deploy/vastai/bootstrap.sh, see deploy/ubuntu/README.md)"
    ;;
  shell) exec docker exec -it -w "$IN_REPO" "$NAME" bash ;;
  run) shift; [ $# -gt 0 ] || die "usage: container.sh run COMMAND..."; exec docker exec -w "$IN_REPO" "$NAME" "$@" ;;
  down) docker rm -f "$NAME" >/dev/null && echo "removed $NAME (kept $WORK)" ;;
  *) die "usage: container.sh [up|shell|run CMD|down]" ;;
esac
