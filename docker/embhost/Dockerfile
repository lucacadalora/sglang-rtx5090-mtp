# local/sglang:v0.5.20-cu130-mtpfix + a backport of open PR #37826 (https://github.com/sgl-project/sglang/pull/37826):
# SGLANG_OFFLOAD_EMBEDDING_TO_HOST=1 keeps the input-embedding table (not the LM head) in pinned host memory and
# gathers rows over PCIe through UVA with a small Triton kernel (CUDA-graph safe, bit-exact). Nothing changes unless
# the env var is set. The patch anchors on exact v0.5.20 source text, so a different base image fails the build.
# Build (CPU only):  docker build -t local/sglang:v0.5.20-cu130-mtpfix-embhost docker/embhost
FROM local/sglang:v0.5.20-cu130-mtpfix
COPY host_embedding_gather.py /tmp/host_embedding_gather.py
COPY apply_embed_offload.py /tmp/apply_embed_offload.py
COPY test_host_gather_wsl.py /opt/embed_offload/test_host_gather_wsl.py
RUN python3 /tmp/apply_embed_offload.py \
 && python3 -c "import sglang.srt.layers.vocab_parallel_embedding as v; assert hasattr(v, 'offload_input_embeddings_to_host'); import sglang.kernels.ops.embeddings.host_embedding_gather as g; print('import OK')" \
 && python3 -m py_compile /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py \
 && rm /tmp/apply_embed_offload.py /tmp/host_embedding_gather.py
