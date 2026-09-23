# Image layer k1 = local/sglang:v0.5.20-cu130-mtpfix-embhost + the kernel-plan patches, every one ENV-GATED and
# DEFAULT OFF (with none of the env vars set the server runs the same kernels and graph shapes as the base image):
#   p1 rank 1  SGLANG_GDN_BA_TINY_GEMM=1 (+ _MAX_M, default 4; _NSPLIT)   in_proj_ba -> tiny_gemm_bf16 at M<=4
#   p2 rank 2  SGLANG_KV_SKIP_UNIT_SCALE_DIV=1                           skip the FP8 KV div_ by unit scales
#   p3 rank 3  SGLANG_AUTOTUNE_TARGET_LMHEAD=1, SGLANG_LMHEAD_FORCE_TACTIC=<t> (+ _MAX_M)  target lm_head autotune / pin
#   p7 rank 7  SGLANG_ATTN_MLP_SILU_FP4_FUSION=1                          SiLU+mul+FP4 fusion in attention-layer MLPs
#   p6 rank 6  SGLANG_EAGLE_SYNCFREE_SEAM=1                               sync-free MTP seam (separate builder);
#              with it on, a host-vs-device layout guard asserts on the first 8 replays per kind and bs by default
#              (_SEAM_CHECK=1 -> first _CHECK_N, default 1000; _SEAM_CHECK=0 -> off)
# Patch order is fixed: p1, p2, p3, p7, p6. Each script asserts exactly one match per anchor, so a changed upstream
# file fails the build. k1_check.py then byte-compiles and imports every modified module and checks the env gates.
# Build (CPU only, no GPU needed):
#   docker build -t local/sglang:v0.5.20-cu130-mtpfix-embhost-k1 docker/k1      (or: bash docker/build.sh)
FROM local/sglang:v0.5.20-cu130-mtpfix-embhost
COPY patches/ /tmp/k1_patches/
COPY k1_check.py /tmp/k1_check.py
RUN set -e; \
    touch /tmp/k1_stamp; sleep 1; \
    python3 /tmp/k1_patches/p1_ba_tiny_gemm.py; \
    python3 /tmp/k1_patches/p2_kv_unit_scale.py; \
    python3 /tmp/k1_patches/p3_lmhead_autotune.py; \
    python3 /tmp/k1_patches/p7_silu_fp4_attn_mlp.py; \
    set -- /tmp/k1_patches/p6_*.py; \
    if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then echo "k1: expected exactly one patches/p6_*.py"; exit 1; fi; \
    python3 "$1"; \
    python3 /tmp/k1_check.py /tmp/k1_stamp; \
    rm -rf /tmp/k1_patches /tmp/k1_check.py /tmp/k1_stamp
