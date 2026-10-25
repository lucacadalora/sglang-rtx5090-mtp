# Run it on your own Ubuntu server with one RTX 5090

The same stack as at home (k1 + k2 patches, MTP head, host-RAM embedding, HiCache, 7 requests at once), on a
headless Ubuntu box: no Windows desktop on the card, no WDDM paging, the whole 32 GB for the model. It runs inside the
stock `lmsysorg/sglang:v0.5.20-cu130` image, patched in place by [`deploy/vastai`](../vastai), so the software is
byte-for-byte what we measured. This folder adds the container wrapper and an A/B matrix for the DFlash2 drafter.

| step | command | time |
|---|---|---|
| 1. Docker + GPU access (once, needs sudo) | see below | ~5 min |
| 2. start the container | `bash deploy/ubuntu/container.sh up` | ~10 min (image pull) |
| 3. patches + weights | `MODELS="27b-mtp 27b-dflash" bash deploy/vastai/bootstrap.sh` | ~10-20 min (25 GB) |
| 4. serve + check | `bash deploy/vastai/serve.sh && bash deploy/vastai/check.sh` | ~5 min |
| 5. A/B matrix | `bash deploy/ubuntu/ab_matrix.sh` | ~45-60 min |

Steps 3-5 run inside the container (`bash deploy/ubuntu/container.sh shell`).

## 1. Docker with GPU access

Check the driver first: `nvidia-smi` must show the RTX 5090 with a driver of **580 or newer** (the image is CUDA 13).
Then install Docker and register the NVIDIA runtime (Ubuntu 24.04; the NVIDIA Container Toolkit must already be
installed, `nvidia-ctk --version`):

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker "$USER"     # then log out and back in
docker run --rm --gpus all lmsysorg/sglang:v0.5.20-cu130 nvidia-smi -L   # must list the RTX 5090
```

## 2. Start the container

```bash
git clone https://github.com/lucacadalora/sglang-rtx5090-mtp && cd sglang-rtx5090-mtp
bash deploy/ubuntu/container.sh up
bash deploy/ubuntu/container.sh shell
```

`up` creates `sglang-5090` from the stock image with the GPU, `--ipc=host`, a 16 GB `/dev/shm`, unlimited memlock
(pinned host-RAM tables) and the host's network, mounts `$WORK` (default `~/sglang-work`) at `/workspace`, and clones
this repo to `/workspace/sglang-rtx5090-mtp`. Weights (`/workspace/hf`), keys, the chat template and logs
(`/workspace/sglang`) live on the host disk, so `container.sh down` + `up` keeps them; re-run `bootstrap.sh` after a
re-create (the patch step is quick, downloads are skipped).

## 3-4. Set up, serve, check

Inside the container:

```bash
MODELS="27b-mtp 27b-dflash" bash deploy/vastai/bootstrap.sh   # patches, 27B + MTP graft, both DFlash2 drafters
bash deploy/vastai/serve.sh                                     # PROFILE=27b-mtp, the home production config
bash deploy/vastai/check.sh                                     # gates + speed at 1, 4 and 7 streams
```

The server binds `127.0.0.1:30000` on the host. From your laptop: `ssh -L 30000:127.0.0.1:30000 <you>@<server>`,
then base URL `http://127.0.0.1:30000/v1`, model `qwen3.8-27b`, key from `/workspace/sglang/api_key` (inside the
container) = `~/sglang-work/sglang/api_key` (on the host). Never publish the port.

Profiles (`PROFILE=... bash deploy/vastai/serve.sh`; stop the running one first with `serve.sh stop`):

| profile | what | at once | notes |
|---|---|---|---|
| `27b-mtp` | MTP head, EAGLE 3/1/4 + ReplaySSM, k1 switches | 7 | home production config |
| `27b-dflash` | DFlash2 drafter, 8-token blocks, 2,048-token draft window | 3 | `DFLASH_DRAFT=nvfp4` (default) or `bf16` |
| `27b` | no speculation (rollback) | 5 | |
| `moe` | Qwen3.6-35B-A3B NVFP4 | 8 | needs `MODELS=moe` at bootstrap |

`HICACHE_GB=max` adds the host-RAM KV tier to `27b-mtp` (and `27b-dflash`, where it allocates and serves but the
restore test has not been run yet). `MAX_RUNNING`, `CONTEXT_LENGTH`, `MEM_FRACTION` override the sizing.

## 5. The A/B matrix

```bash
bash deploy/ubuntu/ab_matrix.sh                                       # all arms, ~45-60 min
ARMS="mtp:7 dflash-bf16:1 dflash-nvfp4:3" bash deploy/ubuntu/ab_matrix.sh   # a subset
WORKLOADS=code TEMP=0 ARMS="dflash-bf16:1" bash deploy/ubuntu/ab_matrix.sh  # the single-stream record attempt
```

Each arm restarts the server with its settings, sends one warm-up reply, runs the quick gates, then measures the
server's own decode throughput and the mean accepted length per verify step for prose, code and structured (JSON)
replies at 1, 3 and 7 streams (those the arm allows), at temperature 0 by default. It writes
`/workspace/sglang/ab/matrix-<stamp>.jsonl` and a `.txt` log with each arm's boot facts (KV pool, window, verify
scratch, free VRAM), prints a summary table, and restarts `27b-mtp` at the end (`RESTORE=0` keeps the last arm).

What the arms test, and why the box matters:

- **`mtp:7`** is the production baseline. At home it measured 131-136 tok/s on one stream (benchmark median; the
  scheduler's steady state is 154-163), 609-627 at 5 and 784-805 at 7 streams, on prose at temperature 0.
- **`dflash-*`**: at home, with a Windows desktop holding ~4 GB of the card, the NVFP4 drafter at 3 running gave 145
  tok/s on one stream (157 by the client) with a 58K-token pool. The blocker is VRAM, not speed: v0.5.20 keeps a
  bf16 GDN state per draft token per running request during verify (2.25 GB at 3 running, 4.5 GB at 7; see
  [docs/DFLASH.md](../../docs/DFLASH.md)). A headless card has ~6 GB more for the model: serve.sh estimates the full
  147,456 window for the NVFP4 drafter at 3 running (~160K pool), ~100K at 7 running, and ~136K for the bf16
  drafter at 1 running. It prints its estimate and shrinks the window when needed; `MEM_FRACTION` can go higher,
  since the default leaves DFLASH more slack than it uses.
- **`dflash-bf16:1`** is SGLang's own one-GPU recipe shape (bf16 drafter, one request at a time). Their cookbook
  reports 4.92 ms per token (~203 tok/s) at an accepted length of 4.29 on an RTX 5090. The bf16 drafter also keeps
  the fused KV-materialisation path that the NVFP4 conversion disables.

Compare like with like: drafters accept far more on code and structured output than on prose (published MTP numbers
on this model range from ~133 tok/s for stories to ~232 for JSON), so read each workload column separately.

## Notes

- Decode is bound by memory bandwidth (14.5 GB read per step), not power: expect ~200-350 W during decode whatever the
  card's power limit is. A memory overclock helps decode, a core overclock mostly helps prefill.
- Long-prompt greedy output can differ between launches and between speculative methods at near-tie tokens; the
  quick gates and `bench/tok_equal.py` (compare within one launch) are the quality check, not byte equality with the
  home reference.
- `serve.sh logs` shows the server log without the line that prints the API keys; `serve.sh stop` frees the GPU.
