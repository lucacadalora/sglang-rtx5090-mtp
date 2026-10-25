# Run it on a rented RTX 5090 (vast.ai)

Three scripts reproduce the server we run at home on any rented RTX 5090 that starts from the stock
`lmsysorg/sglang:v0.5.20-cu130` image. No Docker build and no image push: the instance *is* that image, and the
bootstrap patches SGLang inside it the same way the layers in [`docker/`](../../docker) do.

| step | command | time |
|---|---|---|
| set up once | `bash deploy/vastai/bootstrap.sh` | ~5-15 min (19 GB of weights) |
| start | `bash deploy/vastai/serve.sh` | 2-5 min first time, ~1 min after |
| test | `bash deploy/vastai/check.sh` | ~2 min |
| stop | `bash deploy/vastai/serve.sh stop` | seconds |

## 1. Rent the right machine

The image is CUDA 13, so the host driver must be **580 or newer**. On vast.ai that is the "Max CUDA" column:
filter for **13.0 or higher**, or the instance never starts.

| filter | value | why |
|---|---|---|
| GPU | 1x RTX 5090 | the NVFP4 weights and the k1 kernels are Blackwell (sm_120) work |
| Max CUDA | >= 13.0 | driver 580+ for the cu130 image |
| Disk | >= 150 GB | image ~52 GB unpacked, 27B ~19 GB, MoE ~24 GB, caches |
| RAM | >= 32 GB, 64+ for the host-RAM cache | the server takes ~9 GB; `HICACHE_GB` uses the rest |
| Download speed | the faster the better | the image and the weights are large |

With the [vast CLI](https://cloud.vast.ai/cli/) (check field names with `vastai search offers --help`):

```bash
vastai search offers 'gpu_name=RTX_5090 num_gpus=1 cuda_vers>=13.0 disk_space>=150 reliability>0.98' -o 'dph'
vastai create instance <OFFER_ID> --image lmsysorg/sglang:v0.5.20-cu130 --disk 150 --ssh --direct
```

In the web console: create a template with image `lmsysorg/sglang:v0.5.20-cu130`, launch mode **SSH**, disk 150 GB,
no ports opened. Add your SSH public key under Account first.

## 2. Log in with a tunnel, then set up

Copy the SSH command from the instance's **Connect** button and add the tunnel for the API:

```bash
ssh -p <PORT> root@<HOST> -L 30000:127.0.0.1:30000
```

On the instance:

```bash
cd /workspace && git clone https://github.com/lucacadalora/sglang-rtx5090-mtp && cd sglang-rtx5090-mtp
bash deploy/vastai/bootstrap.sh                       # the 27B with MTP
MODELS="27b-mtp moe" bash deploy/vastai/bootstrap.sh  # or both models
```

What it does, each step skipped when already done: checks the GPU, driver, SGLang version and disk; patches SGLang
in place (mtpfix -> embhost -> k1 -> k2, the same anchored scripts as the Docker layers, every k1/k2 change switched
off until `serve.sh` turns it on); downloads the pinned weight revisions; grafts the MTP head onto the gittensor
NVFP4 checkpoint; saves Qwen's official chat template (sha256 checked); creates the API keys. Everything lands in
`/workspace/hf` (weights) and `/workspace/sglang` (keys, template, logs, kernel caches).

## 3. Start, test, use

```bash
bash deploy/vastai/serve.sh          # PROFILE=27b-mtp: 7 at once, 147,456-token window
bash deploy/vastai/check.sh          # gates + speed at 1, 4 and 7 streams
cat /workspace/sglang/api_key        # the key clients need
```

On your own computer, with the SSH session above open, any OpenAI-compatible client works:

```
base URL  http://127.0.0.1:30000/v1
API key   (the file above)
model     qwen3.8-27b        (qwen3.6-35b-a3b with PROFILE=moe)
```

The server binds 127.0.0.1 only. Do not open its port on the instance: SGLang is not built to face the internet.

| profile | what runs | at once | window |
|---|---|---|---|
| `27b-mtp` (default) | Qwen3.8-27B NVFP4 + grafted MTP head, embedding in host RAM, k1 switches | 7 | 147,456 |
| `27b` | same weights without MTP (the rollback) | 5 | 147,456 |
| `27b-dflash` | same target with the DFlash2 drafter (`DFLASH_DRAFT=nvfp4` or `bf16`; needs `MODELS=27b-dflash` at bootstrap) | 3 | what fits ([docs/DFLASH.md](../../docs/DFLASH.md)) |
| `moe` | Qwen3.6-35B-A3B NVFP4, flashinfer_cutlass MoE runner | 8 | 131,072 |

Switch with `bash deploy/vastai/serve.sh stop && PROFILE=moe bash deploy/vastai/serve.sh` (one model per GPU).

Knobs (environment variables for `serve.sh`):

- `MAX_RUNNING`: requests at once; state slots follow automatically. The MoE took 16 at home (`MAX_RUNNING=16`).
- `CONTEXT_LENGTH`: window. A rented card has no desktop holding VRAM, so the pool is larger than at home; the
  script prints its estimate and shrinks the window if the pool is too small. Above 147,456 is untested by us.
- `HICACHE_GB=max` (27b-mtp only): host-RAM KV tier from k2. An evicted 40K-token conversation came back in
  0.13 s instead of a 5 s re-read. Sized from the container's RAM limit, not the host's.
- `K1=0`: the k1 switches off. `MEM_FRACTION`, `MAMBA_SLOTS`, `EXTRA_ARGS` for anything else.
- `DFLASH_DRAFT`, `DFLASH_BLOCK` (8), `DFLASH_WINDOW` (2048) for `27b-dflash`. Its verify step keeps a GDN state per
  draft token per running request (~73 MiB x 8 x (MAX_RUNNING + 1)); the pool estimate includes it.
- `speed.py --workload prose|code|structured|mixed --temperature 0`: drafters accept far more on code and JSON than on
  prose, so compare speculative setups per workload. The `accept` column is the mean accepted length per step.

## 4. What to expect

Measured on our RTX 5090 under WSL2 with a Windows desktop on the same card (`docs/REPORT.md`). A rented card
has no desktop on it, so it should match or beat these.

| | 27b-mtp | moe |
|---|---|---|
| one user, decode | 150-164 tok/s | 262-303 tok/s |
| all slots busy | 871-891 tok/s (7 streams) | 2,097 tok/s (16 streams) |
| decode at ~61K / ~129K context | 117-121 / 92.5 tok/s | 230 / 201 tok/s |
| prefill 8K | 9.2-12.4K tok/s | 25.6K tok/s |

`check.sh` prints the server's own "gen throughput" next to the client numbers; trust the server column (the GPU
sleeps when idle and needs a moment to ramp up, so client figures read 10-15% low).

## 5. When something goes wrong

- **Instance never starts / "unsatisfied condition: cuda>=13.0"**: the host driver is older than 580. Rent another
  offer with Max CUDA >= 13.0.
- **`python3 cannot import sglang`**: the instance runs another image. vast.ai SSH logins do not always carry the
  image's PATH; `env.sh` restores it from PID 1, so use the scripts rather than calling `sglang` by hand.
- **Out of memory at start**: lower `MAX_RUNNING`, `MAMBA_SLOTS` or `CONTEXT_LENGTH`, or set `MEM_FRACTION=0.85`.
- **Start fails with HiCache on**: the container's RAM share is smaller than it looked. Use a smaller `HICACHE_GB`
  or leave it off.
- **Server log**: `bash deploy/vastai/serve.sh logs` (the line with both API keys is filtered out).
- **Stop paying**: destroy the instance in the console. A stopped instance still bills its disk.

## Not for H100/H200

The bootstrap refuses any GPU that is not compute capability 12.0. Hopper has no FP4 tensor cores, so the NVFP4
weights and the k1 kernels do not apply there; an H100/H200 should serve Qwen's official FP8 or BF16 checkpoints
(which ship their own MTP head) with stock SGLang.
