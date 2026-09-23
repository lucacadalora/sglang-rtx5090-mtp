# SGLang v0.5.20 + open PR #37155 (https://github.com/sgl-project/sglang/pull/37155): the MTP draft shares the
# target's embedding and LM head BEFORE the KV pool is sized. Without it, the draft's duplicate vocab-sized tables
# (4.7 GiB at TP=1 for a 248K vocab) are released only after sizing and never reach the KV cache (issue #36452).
# Build (CPU only):  docker build -t local/sglang:v0.5.20-cu130-mtpfix docker/mtpfix
FROM lmsysorg/sglang:v0.5.20-cu130
COPY pr37155.diff /tmp/pr37155.diff
RUN cd /sgl-workspace/sglang && patch -p1 < /tmp/pr37155.diff && rm /tmp/pr37155.diff \
 && python3 -c "import sglang.srt.speculative.base_spec_worker as m; assert hasattr(m.EagleDraftWorkerBase, 'share_target_embed_and_head')"
