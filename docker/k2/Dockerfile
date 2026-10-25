# Image layer k2 = local/sglang:v0.5.20-cu130-mtpfix-embhost-k1 + the HiCache patches (host-RAM KV tier), every one
# ENV-GATED and DEFAULT OFF (with none of the env vars set the server behaves exactly like k1):
#   p8   SGLANG_HICACHE_HOST_ALLOC=cuda            HiCache host pools from an exact cudaHostAlloc. This is what makes
#                                                  HiCache work under WSL2 (the stock mmap + cudaHostRegister pools
#                                                  hand raw host addresses to the GPU there); also fine on native Linux
#   p9   SGLANG_OFFLOAD_EMBEDDING_EXACT_PIN=1      exact-size pinned embedding table (torch rounds 2.37 -> 4 GiB)
#   p10  SGLANG_HICACHE_MAMBA_SPLIT_BYTES=<n>      split GDN state rows (layer_first path only; the hybrid model must
#                                                  run page_first, where the mamba JIT kernels already use 32 warps)
#   p11  SGLANG_HICACHE_NO_STAGED_WRITE_BACK=1     one-kernel page_first KV backup instead of the staged path, which
#                                                  at page_size 1 issues one cudaMemcpyAsync per token for K and V
#   p12  SGLANG_HICACHE_HOST_RESERVE_GB=<n>        host RAM SGLang keeps free when sizing HiCache pools (upstream 10)
# Served with: --enable-hierarchical-cache --hicache-size <GB> --hicache-mem-layout page_first --hicache-io-backend
# kernel --hicache-write-policy write_through, plus SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 and 7 running requests
# (measured: 871-891 tok/s with 7 streams; an evicted 40K-token conversation restored in 0.13 s vs 5 s re-read).
# Patch order is fixed: p8, p9 (needs p8's exact_pinned.py), p10, p11, p12. Each script asserts its anchor count.
# Build (CPU only, no GPU needed):  docker build -t local/sglang:v0.5.20-cu130-mtpfix-embhost-k2 docker/k2
FROM local/sglang:v0.5.20-cu130-mtpfix-embhost-k1
COPY patches/ /tmp/k2_patches/
COPY k2_check.py /tmp/k2_check.py
RUN set -e; \
    python3 /tmp/k2_patches/p8_hicache_cuda_host_alloc.py; \
    python3 /tmp/k2_patches/p9_embed_exact_pin.py; \
    python3 /tmp/k2_patches/p10_mamba_split_copy.py; \
    python3 /tmp/k2_patches/p11_no_staged_write_back.py; \
    python3 /tmp/k2_patches/p12_host_reserve.py; \
    python3 /tmp/k2_check.py; \
    rm -rf /tmp/k2_patches /tmp/k2_check.py
