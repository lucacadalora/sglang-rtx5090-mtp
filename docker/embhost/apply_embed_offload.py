#!/usr/bin/env python3
"""Backport of sgl-project/sglang PR #37826 (--offload-embedding-to-host) into v0.5.20, gated by an env var
instead of a server arg: SGLANG_OFFLOAD_EMBEDDING_TO_HOST=1 keeps the input-embedding table (not the LM head) in
pinned host memory and gathers rows from the GPU through UVA (CUDA-graph safe, bit-exact).

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server.
"""
import pathlib, shutil

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
VPE = ROOT / "srt/layers/vocab_parallel_embedding.py"
MR = ROOT / "srt/model_executor/model_runner.py"
GATHER_SRC = pathlib.Path("/tmp/host_embedding_gather.py")
GATHER_DST = ROOT / "kernels/ops/embeddings/host_embedding_gather.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"{what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


shutil.copyfile(GATHER_SRC, GATHER_DST)

t = VPE.read_text()
t = replace_once(t, "import logging\n", "import contextlib\nimport logging\nimport os\n", "imports")
t = replace_once(
    t,
    "from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (",
    "from sglang.kernels.ops.embeddings.host_embedding_gather import host_embedding_gather\n"
    "from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (",
    "gather import",
)
t = replace_once(
    t,
    "DEFAULT_VOCAB_PADDING_SIZE = 64\n",
    '''DEFAULT_VOCAB_PADDING_SIZE = 64

# --- backport of PR #37826 (env-gated) ---
# Tables below this size stay on the device when host offload is requested.
HOST_OFFLOAD_MIN_BYTES = 64 << 20


def _host_offload_requested() -> bool:
    return os.environ.get("SGLANG_OFFLOAD_EMBEDDING_TO_HOST", "0").strip().lower() in ("1", "true", "yes")


def _config_ties_word_embeddings() -> bool:
    """True when the served model ties its embedding to the LM head (moving it would break logits)."""
    try:
        from sglang.srt.runtime_context import get_server_args

        hf_config = get_server_args().get_model_config().hf_config
        cfgs = [hf_config, getattr(hf_config, "text_config", None)]
        return any(bool(getattr(c, "tie_word_embeddings", False)) for c in cfgs if c is not None)
    except Exception:
        return False
''',
    "helpers",
)
t = replace_once(
    t,
    """        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
""",
    """        # PR #37826 backport: build an input-embedding table straight in pinned host memory, so the device
        # never allocates it and the free-memory measurement that sizes the KV pool sees the saving.
        _elt = torch.empty((), dtype=params_dtype, device="cpu").element_size() if params_dtype is not None else 0
        place_on_host = (
            is_embedding_layer
            and _host_offload_requested()
            and isinstance(self.quant_method, UnquantizedEmbeddingMethod)
            and params_dtype in (torch.float32, torch.float16, torch.bfloat16)
            and self.num_embeddings_per_partition * self.embedding_dim * _elt >= HOST_OFFLOAD_MIN_BYTES
            and torch.empty(()).device.type == "cuda"
            and not _config_ties_word_embeddings()
        )
        with torch.device("cpu") if place_on_host else contextlib.nullcontext():
            self.quant_method.create_weights(
                self,
                self.embedding_dim,
                [self.num_embeddings_per_partition],
                self.embedding_dim,
                self.num_embeddings_padded,
                params_dtype=params_dtype,
                weight_loader=self.weight_loader,
            )
        if place_on_host:
            self._pin_host_weight()
""",
    "create_weights",
)
t = replace_once(
    t,
    "    def _embed_local_shard(self, input_: torch.Tensor) -> torch.Tensor:\n",
    '''    def _pin_host_weight(self) -> None:
        weight = self.weight
        assert weight.device.type == "cpu"
        self.weight.data = weight.data.pin_memory()
        self._host_weight_verified = True
        logger.info(
            "Embedding table %s created in pinned host memory (%.2f GiB)",
            type(self).__name__, weight.numel() * weight.element_size() / (1 << 30),
        )

    def offload_weight_to_host(self) -> int:
        weight = self.weight
        if not weight.is_cuda:
            return 0
        if weight.numel() * weight.element_size() < HOST_OFFLOAD_MIN_BYTES:
            return 0
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            logger.warning("Not offloading %s to host: quantized embedding %s", type(self).__name__,
                           type(self.quant_method).__name__)
            return 0
        if weight.ndim != 2 or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return 0
        host = torch.empty(weight.shape, dtype=weight.dtype, device="cpu", pin_memory=True)
        host.copy_(weight)
        nbytes = weight.numel() * weight.element_size()
        self.weight.data = host
        self._host_weight_verified = True
        logger.info("Embedding table %s moved to pinned host memory (%.2f GiB)", type(self).__name__,
                    nbytes / (1 << 30))
        return nbytes

    def _weight_is_host_resident(self, input_: torch.Tensor) -> bool:
        if self.weight.device.type != "cpu" or not input_.is_cuda:
            return False
        if not getattr(self, "_host_weight_verified", False):
            # Can arrive through weight sharing (a draft handed the target's table).
            if not self.weight.is_pinned():
                raise RuntimeError("VocabParallelEmbedding weight is on the host but not pinned")
            self._host_weight_verified = True
        return True

    def _embed_local_shard_from_host(self, input_: torch.Tensor, symm_alloc) -> torch.Tensor:
        if self.tp_size == 1:
            with symm_alloc:
                output_parallel = host_embedding_gather(input_, self.weight)
        else:
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            with symm_alloc:
                output_parallel = host_embedding_gather(masked_input, self.weight)
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        if self.output_dtype is not None:
            output_parallel = output_parallel.to(self.output_dtype)
        return output_parallel

    def _embed_local_shard(self, input_: torch.Tensor) -> torch.Tensor:
''',
    "methods",
)
t = replace_once(
    t,
    """        symm_alloc = use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        )
        if self.tp_size == 1:
            with symm_alloc:
                output_parallel = self.quant_method.embedding(self, input_.long())""",
    """        symm_alloc = use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        )
        if self._weight_is_host_resident(input_):
            return self._embed_local_shard_from_host(input_, symm_alloc)
        if self.tp_size == 1:
            with symm_alloc:
                output_parallel = self.quant_method.embedding(self, input_.long())""",
    "routing",
)
t = replace_once(
    t,
    "class ParallelLMHead(VocabParallelEmbedding):\n",
    '''def offload_input_embeddings_to_host(model: torch.nn.Module) -> List[Tuple[str, int]]:
    """Post-load fallback: move any input-embedding table still on the device (not LM heads, not tied)."""
    lm_head_ptrs = set()
    for module in model.modules():
        if isinstance(module, ParallelLMHead):
            w = getattr(module, "weight", None)
            if w is not None:
                lm_head_ptrs.add(w.data.data_ptr())
    moved = []
    for name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding) or isinstance(module, ParallelLMHead):
            continue
        if module.weight.data.data_ptr() in lm_head_ptrs:
            logger.info("Not offloading %s: weight is tied with an LM head", name)
            continue
        nbytes = module.offload_weight_to_host()
        if nbytes:
            moved.append((name, nbytes))
    return moved


class ParallelLMHead(VocabParallelEmbedding):
''',
    "module fn",
)
assert "get_masked_input_and_mask" in t, "get_masked_input_and_mask missing in v0.5.20"
VPE.write_text(t)

m = MR.read_text()
m = replace_once(
    m,
    """        if not self.is_draft_worker:
            get_offloader().post_init()
""",
    """        if not self.is_draft_worker:
            get_offloader().post_init()
            # PR #37826 backport: post-load fallback for tables built on a meta device.
            import os as _os
            if _os.environ.get("SGLANG_OFFLOAD_EMBEDDING_TO_HOST", "0").strip().lower() in ("1", "true", "yes"):
                from sglang.srt.layers.vocab_parallel_embedding import (
                    ParallelLMHead as _PLH,
                    VocabParallelEmbedding as _VPE,
                    offload_input_embeddings_to_host as _offload,
                )
                _moved = _offload(self.model)
                if _moved:
                    torch.cuda.empty_cache()
                for _n, _b in _moved:
                    logger.info(f"Offloaded input embedding {_n} ({_b / (1 << 30):.2f} GB) to pinned host memory after loading")
                _on_host = [n for n, mod in self.model.named_modules()
                            if isinstance(mod, _VPE) and not isinstance(mod, _PLH) and mod.weight.device.type == "cpu"]
                logger.info(f"Input embeddings in pinned host memory: {_on_host or 'NONE'}")
""",
    "model_runner hook",
)
MR.write_text(m)
print("embed-offload backport applied")
