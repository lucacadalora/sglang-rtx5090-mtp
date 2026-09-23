#!/usr/bin/env python3
"""k1 p1 (plan rank 1): route the GDN in_proj_ba bf16 GEMM to sglang's JIT tiny_gemm_bf16 for tiny M.

Why: in_proj_ba is a bf16 [96, 5120] projection (in_proj_a/in_proj_b are excluded from NVFP4). At M=2..4 tokens
cuBLAS 13.0 on SM120 picks cutlass_80_wmma 16x16_128x1 (6 useful one-warp CTAs, no split-K): 35-47 us per call,
0.87 ms/cycle exposed at 1-stream MTP verify (M=4) because the main stream joins on it after the qkvz GEMM.
tiny_gemm_bf16 (kernels/ops/gemm/tiny_gemm.py, already in the image) does it in ~2 us.

Env (read once per GDN layer, at __init__ time; all unset = today's code path, same kernels, same graph shapes):
  SGLANG_GDN_BA_TINY_GEMM=1            enable (default off).
  SGLANG_GDN_BA_TINY_GEMM_MAX_M=<int>  largest M routed to tiny_gemm (default 4 when enabled; M=5+ stays cuBLAS,
                                       which is already split-K and hidden there). Also the kernel's compiled max_m.
  SGLANG_GDN_BA_TINY_GEMM_NSPLIT=<int> output rows per CTA (default: unset = the kernel's own default, n_split=1,
                                       96 CTAs for N=96, which is what ba_proj bench4a/4b/5 verified). 3 -> 32 CTAs,
                                       4 -> 24 CTAs (leaves SMs for the 128-CTA qkvz NVFP4 GEMM). Must divide N and
                                       n_split*MAX_M <= K/16, else the default is used (logged once).

Guards (any miss falls back to the unchanged cuBLAS path, self.in_proj_ba(x)): 0 < M <= MAX_M; 2-D bf16 x and
weight; x.stride(1) == 1; x.stride(0) % 16 == 0 (sm_120 CUDA>=12.9 loads 32-byte vectors, tiny_gemm.cuh CHECK_HOST
needs stride % kVecSize=16); x.data_ptr() % 32 == 0; weight contiguous and 32-byte aligned; no bias;
UnquantizedLinearMethod; not a LoRA wrapper; not under torch.compile tracing; can_use_tiny_gemm(N, K, MAX_M).
Used at both call sites of Qwen3_5GatedDeltaNet._forward_input_proj: the alt-stream capture branch (the fork/join
is kept as is) and the plain branch.

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server.
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
QWEN = ROOT / "srt/models/qwen3_5.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p1 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


t = QWEN.read_text()
assert "_gdn_ba_tiny_gemm" not in t, "p1 already applied"

# 1) module-level config reader + guarded dispatcher, placed just before the GDN class
t = replace_once(
    t,
    "class Qwen3_5GatedDeltaNet(nn.Module):\n",
    '''# --- k1 p1: GDN in_proj_ba -> tiny_gemm_bf16 for tiny M (env-gated, default off) ---
_GDN_BA_TINY_GEMM_LOGGED: set = set()


def _gdn_ba_tiny_gemm_config() -> Tuple[int, Optional[int]]:
    """(max_m, n_split) from SGLANG_GDN_BA_TINY_GEMM*; (0, None) = off (today's cuBLAS path)."""
    if os.environ.get("SGLANG_GDN_BA_TINY_GEMM", "0").strip().lower() not in ("1", "true", "yes"):
        return 0, None
    if not _is_cuda:
        return 0, None
    try:
        max_m = int(os.environ.get("SGLANG_GDN_BA_TINY_GEMM_MAX_M", "4"))
    except ValueError:
        logger.warning("SGLANG_GDN_BA_TINY_GEMM_MAX_M is not an int; using 4")
        max_m = 4
    if max_m <= 0:
        return 0, None
    n_split: Optional[int] = None
    raw = os.environ.get("SGLANG_GDN_BA_TINY_GEMM_NSPLIT", "").strip()
    if raw:
        try:
            n_split = int(raw)
        except ValueError:
            logger.warning("SGLANG_GDN_BA_TINY_GEMM_NSPLIT is not an int; using the kernel default")
        if n_split is not None and n_split <= 0:
            n_split = None
    if "config" not in _GDN_BA_TINY_GEMM_LOGGED:
        _GDN_BA_TINY_GEMM_LOGGED.add("config")
        logger.info(
            "GDN in_proj_ba: tiny_gemm_bf16 enabled for 0 < M <= %d (n_split=%s)",
            max_m,
            "default" if n_split is None else n_split,
        )
    return max_m, n_split


def _gdn_ba_tiny_gemm(
    x, layer: nn.Module, max_m: int, n_split: Optional[int]
) -> Optional[torch.Tensor]:
    """tiny_gemm_bf16(x, layer.weight) when every guard holds, else None (caller keeps cuBLAS)."""
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        return None
    m = x.shape[0]
    if not (0 < m <= max_m):
        return None
    if torch.compiler.is_compiling():
        return None
    w = getattr(layer, "weight", None)
    if (
        not isinstance(w, torch.Tensor)
        or getattr(layer, "bias", None) is not None
        or hasattr(layer, "base_layer")  # LoRA wrapper
        or not isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod)
        or x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not x.is_cuda
        or x.device != w.device
        or x.shape[1] != w.shape[1]
        or x.stride(1) != 1
        or x.stride(0) % 16 != 0
        or x.data_ptr() % 32 != 0
        or not w.is_contiguous()
        or w.data_ptr() % 32 != 0
    ):
        return None
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

    n, k = w.shape
    if not can_use_tiny_gemm(n, k, max_m):
        return None
    if n_split is not None and (n % n_split != 0 or n_split * max_m > k // 16):
        if ("nsplit", n, k, n_split) not in _GDN_BA_TINY_GEMM_LOGGED:
            _GDN_BA_TINY_GEMM_LOGGED.add(("nsplit", n, k, n_split))
            logger.warning(
                "SGLANG_GDN_BA_TINY_GEMM_NSPLIT=%d invalid for N=%d K=%d max_m=%d; using the kernel default",
                n_split, n, k, max_m,
            )
        n_split = None
    return tiny_gemm_bf16(x, w, max_m=max_m, n_split=n_split)


class Qwen3_5GatedDeltaNet(nn.Module):
''',
    "helpers",
)

# 2) read the env once per layer, right after in_proj_ba is built
t = replace_once(
    t,
    "        self._bind_packed_weight_loaders(self.in_proj_qkvz)\n"
    "        self._bind_packed_weight_loaders(self.in_proj_ba)\n",
    "        self._bind_packed_weight_loaders(self.in_proj_qkvz)\n"
    "        self._bind_packed_weight_loaders(self.in_proj_ba)\n"
    "        # k1 p1: (0, None) unless SGLANG_GDN_BA_TINY_GEMM=1\n"
    "        self._ba_tiny_gemm_max_m, self._ba_tiny_gemm_n_split = _gdn_ba_tiny_gemm_config()\n",
    "init",
)

# 3) the dispatch method, just before _forward_input_proj
t = replace_once(
    t,
    "    def _forward_input_proj(self, hidden_states: torch.Tensor):\n",
    '''    def _in_proj_ba_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # k1 p1: tiny_gemm_bf16 for 0 < M <= SGLANG_GDN_BA_TINY_GEMM_MAX_M, else unchanged.
        max_m = getattr(self, "_ba_tiny_gemm_max_m", 0)
        if max_m > 0:
            out = _gdn_ba_tiny_gemm(
                hidden_states, self.in_proj_ba, max_m, self._ba_tiny_gemm_n_split
            )
            if out is not None:
                return out
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_ba

    def _forward_input_proj(self, hidden_states: torch.Tensor):
''',
    "method",
)

# 4) call site 1: alt-stream capture branch (fork/join kept)
t = replace_once(
    t,
    "            with torch.cuda.stream(self.alt_stream):\n"
    "                projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
    "            current_stream.wait_stream(self.alt_stream)\n"
    "        elif self._fused_input_proj_cpu_enabled.value:\n",
    "            with torch.cuda.stream(self.alt_stream):\n"
    "                projected_states_ba = self._in_proj_ba_forward(hidden_states)\n"
    "            current_stream.wait_stream(self.alt_stream)\n"
    "        elif self._fused_input_proj_cpu_enabled.value:\n",
    "alt-stream call site",
)

# 5) call site 2: plain branch
t = replace_once(
    t,
    "        else:\n"
    "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
    "        return projected_states_qkvz, projected_states_ba\n"
    "\n"
    "    def _forward_input_proj_fused_quant_amd(self, hidden_states):\n",
    "        else:\n"
    "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            projected_states_ba = self._in_proj_ba_forward(hidden_states)\n"
    "        return projected_states_qkvz, projected_states_ba\n"
    "\n"
    "    def _forward_input_proj_fused_quant_amd(self, hidden_states):\n",
    "plain call site",
)

QWEN.write_text(t)
print("p1_ba_tiny_gemm applied")
