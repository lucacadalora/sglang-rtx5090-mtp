#!/usr/bin/env python3
"""k1 patch p6: sync-free MTP seam for EAGLE topk=1 on FlashInfer (plan rank 6 = bubbles Opt1+Opt2+Opt3+Opt5).

ENV-GATED, DEFAULT OFF. With SGLANG_EAGLE_SYNCFREE_SEAM unset (or 0) every patched code path takes the stock
branch: same wrappers, same plan() calls, same kernels, same CUDA-graph shapes, same stream/event usage.

Env (read once per process, at first use, i.e. during model-runner / scheduler init). Booleans follow sglang's
EnvBool rules (environ.py): true/1/yes/y and false/0/no/n, case-insensitive; any other value logs a warning and
counts as unset. The sub-flags below are only read (and validated) when the master switch is on, so a bad
sub-flag can never break a server that runs with the seam off.
  SGLANG_EAGLE_SYNCFREE_SEAM=1         master switch.
  SGLANG_EAGLE_SYNCFREE_SEAM_PARTS     comma list of the parts to enable, default "verify,d2h,draft":
      verify  (Opt1) EAGLE target-verify CUDA-graph wrappers are captured WITHOUT a custom-mask buffer when
              topk == 1 and draft_token_num == num_steps + 1 (the topk=1 tree is a causal chain), so the kernel
              runs MaskMode.CAUSAL (forward(causal=True)), and replays plan with the sync-free fast_prefill_plan
              from host-known layout: qo_indptr = arange(0,(bs+1)*N,N), kv_lens = seq_lens_cpu + N,
              kv_indptr = [0, cumsum(kv_lens)], max_q_len = N, max_kv_len = max(seq_lens_cpu) + N.
              Removes flashinfer plan()'s mask_indptr pageable H2D + stream sync that drained the draft graph,
              the segment_packbits .item() and the three .to("cpu") reads.
      d2h     (Opt2 + Opt5) FutureMap.publish() queues the new_seq_lens D2H on fwd_prepare_d2h_stream right
              behind the publish event and records a done-event; resolve_seq_lens_cpu() waits on that event
              instead of issuing its own copy (which queued behind copy_result's D2H under WDDM) and then
              cudaStreamSynchronize-spinning. _DEBUG_ASSERT (CI) and resolve_mixed_spec_tails keep their own
              copy; the per-cycle resolve wait is never removed (it keeps the pinned plan buffers safe).
      draft   (Opt3) FlashInferMultiStepDraftBackend builds the draft-decode kv_indptr on the host
              ([0, cumsum(positions + i + 1)] with positions = seq_lens_cpu, 0 on graph-padding rows) instead of
              the blocking kv_indptr.cpu(); only on CUDA-graph replays with topk == 1, page_size == 1.
  SGLANG_EAGLE_SYNCFREE_SEAM_WAIT      d2h part, how the scheduler waits for the done-event (Opt5):
      event (default) cudaEventBlockingSync event + synchronize(), as designed. Measured under WSL2 (this box):
                      still spins the calling thread at 100% CPU, so it saves no CPU here; latency as spin.
      spin            plain event + synchronize() (stock behaviour), for A/B.
      poll            sleep 200 us steps while the wait is younger than 0.75 x its running mean - 0.3 ms,
                      polling event.query(), then spin the tail: actually frees the core; adds wake latency
                      only if a wait ends much earlier than usual.
  SGLANG_EAGLE_SYNCFREE_SEAM_CHECK     host-vs-device layout asserts (each one syncs), counted per kind and per
                                       padded graph bs, on real replays only (capture-time dummy inputs never
                                       spend the budget); a failed assert raises and stops the server:
      unset   DEFAULT with the seam on: warm-up guard, the first 8 replays of each kind at each bs.
      1       validation mode: the first SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N replays (default 1000).
      0       off (no layout guard at all; only for A/B timing of the guard itself).
      verify_mask   the replayed EAGLE tree mask equals the causal chain and positions == seq_len + q;
      verify_indptr fast-plan host qo/kv indptr == device qo_indptr / cum_kv_seq_len (eagle_info.py:98-103);
      draft_indptr  host draft kv_indptr == device kv_indptr written by generate_draft_decode_kv_indices.
      Output-level gates cannot replace these asserts: a host kv layout short by N (the DFLASH-branch formula)
      or stale by 1 gives bit-identical attention output unless a split-kv chunk boundary falls in the last few
      tokens, where it silently drops the draft tokens' own KV (reviewer GPU negative control, rv_gpu_e2e.py).
  SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N   override the per-(kind, bs) budget (default 8 when CHECK is unset, 1000
                                       when CHECK=1).

Also, with the seam on, every fast_prefill_plan wrapper (EAGLE verify and draft-extend) gets a per-wrapper plan
fence: the CPU never rewrites a wrapper's pinned plan buffer before the previous upload from it ran on the GPU.
The draft-decode fast_decode_plan pinned buffers (draft part) have no fence of their own. They are safe only by
ordering: in every cycle N the draft(N) plan upload is queued before verify(N), verify(N) before publish_ready(N),
and the scheduler's resolve wait (resolve_seq_lens_cpu, never removed) waits on a D2H queued behind
publish_ready(N) before the CPU plans draft(N+1) and rewrites those buffers. A change that moves draft planning
ahead of that resolve wait, or records publish_ready before the draft upload, would race silently.

Refuses to turn on (warning) when sglang's own envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH is true (any EnvBool
spelling, e.g. 1 or y): the glue graph would freeze the host-fed plans. Validated for single-layer EAGLE/MTP v2
(Qwen3.8-27B + MTP, topk 1, 3 steps, 4 draft tokens) only.

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server. Adds one leaf module:
sglang/srt/speculative/syncfree_seam.py.
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
FIB = ROOT / "srt/layers/attention/flashinfer_backend.py"
OVL = ROOT / "srt/managers/overlap_utils.py"
HELPER = ROOT / "srt/speculative/syncfree_seam.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"{what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


HELPER_SRC = r'''"""Sync-free MTP seam helpers (k1 patch p6). Leaf module: imports only os/logging/time/torch at import time
(sglang.srt.environ is imported lazily, only when the seam is switched on).

Everything here is inert unless SGLANG_EAGLE_SYNCFREE_SEAM is true (see the p6 patch docstring for the env list).
With the seam on, the host-vs-device layout asserts run by default on the first WARMUP_CHECK_N real replays of
each kind at each padded graph bs (SGLANG_EAGLE_SYNCFREE_SEAM_CHECK unset); CHECK=1 widens that to
SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N (default 1000), CHECK=0 turns the guard off.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# sglang EnvBool rules (srt/environ.py): case-insensitive, no other spellings.
_TRUE = ("true", "1", "yes", "y")
_FALSE = ("false", "0", "no", "n")
ALL_PARTS = ("verify", "d2h", "draft")

WAIT_MODES = ("event", "spin", "poll")
WARMUP_CHECK_N = 8  # default layout-guard budget per (kind, bs) when SGLANG_EAGLE_SYNCFREE_SEAM_CHECK is unset
FULL_CHECK_N = 1000  # default budget with SGLANG_EAGLE_SYNCFREE_SEAM_CHECK=1

_enabled: Optional[bool] = None
_parts: frozenset = frozenset()
_wait_mode = "event"
_wait_ema = 0.0
_check = False
_check_n = 0
_check_started: dict = {}
_check_passed: dict = {}
_fence_waits = 0
_warned: set = set()


def _env_bool(name: str, default):
    """sglang EnvBool parse: unset -> default; true/1/yes/y -> True; false/0/no/n -> False; else warn, default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning("[p6] invalid boolean %s=%r (EnvBool: true/1/yes/y, false/0/no/n); treated as unset", name, raw)
    return default


def _glue_graph_on() -> bool:
    """sglang's own reading of SGLANG_ENABLE_METADATA_GLUE_GRAPH (decode_cuda_graph_runner.py uses the same)."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get())


def _load() -> None:
    global _enabled, _parts, _wait_mode, _check, _check_n
    on = bool(_env_bool("SGLANG_EAGLE_SYNCFREE_SEAM", False))
    if on and _glue_graph_on():
        logger.warning(
            "[p6] SGLANG_EAGLE_SYNCFREE_SEAM ignored: SGLANG_ENABLE_METADATA_GLUE_GRAPH is on and would freeze "
            "the host-fed verify/draft plans inside the glue graph"
        )
        on = False
    parts = frozenset()
    mode = "event"
    check = False
    check_n = 0
    if on:
        # Sub-flags are parsed (and may raise) only with the master switch on.
        raw = os.environ.get("SGLANG_EAGLE_SYNCFREE_SEAM_PARTS", ",".join(ALL_PARTS))
        parts = frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
        unknown = sorted(parts - set(ALL_PARTS))
        if unknown:
            raise ValueError(
                f"SGLANG_EAGLE_SYNCFREE_SEAM_PARTS has unknown parts {unknown}; allowed: {ALL_PARTS}"
            )
        mode = os.environ.get("SGLANG_EAGLE_SYNCFREE_SEAM_WAIT", "event").strip().lower() or "event"
        if mode not in WAIT_MODES:
            raise ValueError(f"SGLANG_EAGLE_SYNCFREE_SEAM_WAIT={mode!r}; allowed: {WAIT_MODES}")
        # CHECK unset (or not a valid boolean) -> warm-up guard; CHECK=1 -> validation window; CHECK=0 -> off.
        check_env = _env_bool("SGLANG_EAGLE_SYNCFREE_SEAM_CHECK", None)
        if check_env is not False:
            default_n = FULL_CHECK_N if check_env else WARMUP_CHECK_N
            raw_n = os.environ.get("SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N")
            try:
                check_n = default_n if raw_n is None else int(raw_n)
            except ValueError:
                raise ValueError(f"SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N={raw_n!r} is not an integer") from None
            if check_n < 0:
                raise ValueError(f"SGLANG_EAGLE_SYNCFREE_SEAM_CHECK_N={check_n} must be >= 0")
            check = check_n > 0
    _parts = parts
    _wait_mode = mode
    _check = check
    _check_n = check_n
    _enabled = bool(on and parts)
    if _enabled:
        logger.info(
            "[p6] sync-free MTP seam ON: parts=%s d2h_wait=%s layout checks=%s",
            ",".join(p for p in ALL_PARTS if p in parts),
            _wait_mode,
            f"first {_check_n} replays per kind and bs" if _check else "OFF",
        )
        if not _check:
            logger.warning(
                "[p6] host-vs-device layout guard disabled (SGLANG_EAGLE_SYNCFREE_SEAM_CHECK=0 or _CHECK_N=0): "
                "a host layout error would not be visible in the output"
            )


def enabled(part: Optional[str] = None) -> bool:
    """True when the seam (or one of its parts) is switched on for this process."""
    if _enabled is None:
        _load()
    if not _enabled:
        return False
    return part is None or part in _parts


def wait_mode() -> str:
    if _enabled is None:
        _load()
    return _wait_mode


def warn_once(key: str, msg: str, *args) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning("[p6] " + msg, *args)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _check_key(kind: str, bs: Optional[int]):
    return kind if bs is None else (kind, int(bs))


def check_due(kind: str, bs: Optional[int] = None) -> bool:
    """True while the layout guard is on and fewer than _check_n checks of (kind, bs) ran; never during stream
    capture. Callers must only ask on a real replay (capture-time dummy inputs must not spend the budget)."""
    if not enabled() or not _check or _capturing():
        return False
    key = _check_key(kind, bs)
    n = _check_started.get(key, 0)
    if n >= _check_n:
        return False
    _check_started[key] = n + 1
    return True


def _ok(kind: str, bs: Optional[int] = None) -> None:
    key = _check_key(kind, bs)
    n = _check_passed.get(key, 0) + 1
    _check_passed[key] = n
    if n == 1 or n == _check_n:
        if bs is None:
            logger.info("[p6] seam check %s passed (%d/%d)", kind, n, _check_n)
        else:
            logger.info("[p6] seam check %s passed (bs=%d, %d/%d)", kind, int(bs), n, _check_n)


# ---------------------------------------------------------------- Opt1: EAGLE topk=1 verify (flashinfer)


def is_eagle_chain_verify(backend, forward_mode, spec_info) -> bool:
    """Capture-time decision: may this target-verify CUDA graph run causal (no tree mask) + fast plan?"""
    if not enabled("verify"):
        return False
    if not forward_mode.is_target_verify() or spec_info is None:
        return False
    from sglang.srt.speculative.spec_info import SpecInputType

    if getattr(spec_info, "spec_input_type", None) != SpecInputType.EAGLE_VERIFY:
        return False
    n = getattr(spec_info, "draft_token_num", None)
    steps = getattr(spec_info, "spec_steps", None)
    if getattr(spec_info, "topk", None) != 1 or n is None or steps is None or n < 1 or n != steps + 1:
        return False
    if getattr(spec_info, "num_tokens_per_req", n) != n:
        return False
    if getattr(spec_info, "ragged_verify_layout", None) is not None:
        return False
    if (
        getattr(backend, "prefill_backend", None) != "fa2"
        or getattr(backend, "dispatch_reason", "?") is not None
        or getattr(backend, "use_sliding_window_kv_pool", True)
        or getattr(backend, "is_dllm_model", True)
        or getattr(backend, "prefill_uses_dequant_workspace", True)
        or getattr(backend, "page_size", 0) != 1
    ):
        warn_once(
            "verify_backend",
            "verify part not applied: needs fa2 prefill, full attention only, no dLLM, no dequant "
            "workspace, page_size 1",
        )
        return False
    return True


def validate_chain_spec_info(spec_info, n: int) -> None:
    """Every call on a causal-captured verify wrapper must carry a topk=1 chain of the captured width."""
    from sglang.srt.speculative.spec_info import SpecInputType

    if not (
        spec_info is not None
        and getattr(spec_info, "spec_input_type", None) == SpecInputType.EAGLE_VERIFY
        and getattr(spec_info, "topk", None) == 1
        and getattr(spec_info, "draft_token_num", None) == n
        and getattr(spec_info, "spec_steps", None) == n - 1
        and getattr(spec_info, "num_tokens_per_req", None) == n
        and getattr(spec_info, "ragged_verify_layout", None) is None
    ):
        raise RuntimeError(
            f"[p6] causal-captured EAGLE verify wrapper (chain width {n}) got a non-chain spec input: "
            f"type={getattr(spec_info, 'spec_input_type', None)} topk={getattr(spec_info, 'topk', None)} "
            f"draft_token_num={getattr(spec_info, 'draft_token_num', None)} "
            f"spec_steps={getattr(spec_info, 'spec_steps', None)}"
        )


def check_chain_mask(spec_info, seq_lens_cpu, custom_mask, n: int, bs: Optional[int] = None) -> None:
    """The replayed EAGLE tree mask must be the causal chain (prefix all visible, tril over the N tokens)."""
    draft = getattr(spec_info, "draft_token", None)
    if draft is None or custom_mask is None or seq_lens_cpu is None:
        return  # capture-time dummy input (the caller already skips those; kept as a guard)
    n_real = draft.numel() // n
    lens = [int(x) for x in seq_lens_cpu[:n_real].tolist()]
    m = custom_mask.detach().reshape(-1).to("cpu").to(torch.bool)
    tri = torch.tril(torch.ones((n, n), dtype=torch.bool))
    off = 0
    for i, seq_len in enumerate(lens):
        size = n * (seq_len + n)
        if off + size > m.numel():
            raise AssertionError(
                f"[p6] verify mask too short: request {i}/{n_real} needs [{off},{off + size}) of {m.numel()}"
            )
        blk = m[off : off + size].view(n, seq_len + n)
        if not bool(blk[:, :seq_len].all()):
            raise AssertionError(f"[p6] verify mask hides part of the prefix of request {i} (seq_len {seq_len})")
        if not torch.equal(blk[:, seq_len:], tri):
            raise AssertionError(
                f"[p6] verify tree mask of request {i} is not the causal chain: {blk[:, seq_len:].int().tolist()}"
            )
        off += size
    pos = getattr(spec_info, "positions", None)
    if pos is not None and n_real > 0:
        p = pos.detach().reshape(-1)[: n_real * n].to("cpu").to(torch.int64).view(n_real, n)
        exp = torch.tensor(lens, dtype=torch.int64).view(-1, 1) + torch.arange(n, dtype=torch.int64).view(1, -1)
        if not torch.equal(p, exp):
            raise AssertionError(f"[p6] verify positions {p.tolist()} != seq_len + q {exp.tolist()}")
    _ok("verify_mask", bs)


def check_verify_indptr(qo_dev, kv_dev, qo_host, kv_host, bs: int) -> None:
    d_qo = qo_dev.detach()[: bs + 1].to("cpu").to(torch.int64)
    d_kv = kv_dev.detach()[: bs + 1].to("cpu").to(torch.int64)
    h_qo = qo_host.to(torch.int64)
    h_kv = kv_host.to(torch.int64)
    if not (torch.equal(d_qo, h_qo) and torch.equal(d_kv, h_kv)):
        raise AssertionError(
            f"[p6] verify layout host/device mismatch at bs={bs}: qo dev={d_qo.tolist()} host={h_qo.tolist()} "
            f"kv dev={d_kv.tolist()} host={h_kv.tolist()}"
        )
    _ok("verify_indptr", bs)


# ---------------------------------------------------------------- Opt3: host draft-decode kv_indptr


def host_draft_kv_indptr(forward_batch, bs: int, num_steps: int) -> Optional[torch.Tensor]:
    """[num_steps, bs+1] int32: row i = [0, cumsum(positions + i + 1)], the formula of
    generate_draft_decode_kv_indices (cache_locs.py) at topk=1, where positions = seq_lens (0 on padding rows).
    None when the host inputs are missing (caller falls back to the device copy)."""
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    num_padding = getattr(forward_batch, "num_padding", None)
    if seq_lens_cpu is None or num_padding is None or seq_lens_cpu.shape[0] < bs or num_padding > bs:
        return None
    pos = seq_lens_cpu[:bs].to(torch.int64).clone()
    if num_padding > 0:
        pos[bs - int(num_padding) :] = 0
    steps = torch.arange(1, num_steps + 1, dtype=torch.int64).view(-1, 1)
    out = torch.zeros((num_steps, bs + 1), dtype=torch.int32)
    out[:, 1:] = torch.cumsum(pos.view(1, -1) + steps, dim=1).to(torch.int32)
    return out


def check_draft_kv_indptr(host: torch.Tensor, dev: torch.Tensor, forward_batch, bs: int) -> None:
    d = dev.detach().to("cpu").to(torch.int64)
    if not torch.equal(d, host.to(torch.int64)):
        pos = getattr(forward_batch, "positions", None)
        pos_l = pos.detach()[:bs].to("cpu").tolist() if pos is not None else None
        raise AssertionError(
            f"[p6] draft kv_indptr host/device mismatch at bs={bs}: dev={d.tolist()} host={host.tolist()} "
            f"positions={pos_l} seq_lens_cpu={forward_batch.seq_lens_cpu[:bs].tolist()} "
            f"num_padding={getattr(forward_batch, 'num_padding', None)}"
        )
    _ok("draft_indptr", bs)


# ---------------------------------------------------------------- pinned plan-buffer fence (fast_prefill_plan)


def plan_fence_wait(wrapper) -> None:
    """Block until the previous plan upload from this wrapper's pinned buffer ran on the GPU (normally done)."""
    global _fence_waits
    ev = getattr(wrapper, "_p6_plan_done", None)
    if ev is None or ev.query():
        return
    _fence_waits += 1
    if _fence_waits <= 5 or _fence_waits % 1000 == 0:
        logger.warning(
            "[p6] plan fence: CPU reached the next fast plan of a wrapper before its previous upload ran "
            "(%d times so far); waiting",
            _fence_waits,
        )
    ev.synchronize()


def plan_fence_record(wrapper) -> None:
    if not torch.cuda.is_available() or _capturing():
        return
    ev = getattr(wrapper, "_p6_plan_done", None)
    if ev is None:
        ev = torch.cuda.Event()
        wrapper._p6_plan_done = ev
    ev.record()


# ---------------------------------------------------------------- Opt2/Opt5: early seq_lens D2H


def new_d2h_done_event(device):
    """Done-event for the early seq_lens D2H: cudaEventBlockingSync in "event" mode, plain otherwise."""
    return torch.get_device_module(device).Event(blocking=(wait_mode() == "event"))


def wait_d2h(ev) -> None:
    """Wait for the early seq_lens D2H.

    event: ev.synchronize() on a blocking-sync event (the design's Opt5; under WSL2 it was measured to still
           spin the calling thread at 100% CPU).
    spin:  ev.synchronize() on a plain event (stock spin behaviour).
    poll:  sleep in 200 us steps while the wait is younger than 0.75 x its running average (minus 0.3 ms),
           polling ev.query(); then spin for the tail, so the wake-up stays precise when the CPU is ahead.
    """
    global _wait_ema
    if _wait_mode != "poll":
        ev.synchronize()
        return
    t0 = time.perf_counter()
    if not ev.query():
        budget = 0.75 * _wait_ema - 0.0003
        while True:
            left = budget - (time.perf_counter() - t0)
            if left <= 0:
                ev.synchronize()
                break
            time.sleep(min(0.0002, left))
            if ev.query():
                break
    waited = time.perf_counter() - t0
    _wait_ema = waited if _wait_ema == 0.0 else 0.9 * _wait_ema + 0.1 * waited
'''

assert not HELPER.exists(), f"{HELPER} already exists"
HELPER.write_text(HELPER_SRC)

# ------------------------------------------------------------------------------------------ flashinfer_backend.py
t = FIB.read_text()

t = replace_once(
    t,
    "from sglang.srt.speculative.spec_info import SpecInput, SpecInputType\n",
    "from sglang.srt.speculative import syncfree_seam as _p6_seam\n"
    "from sglang.srt.speculative.spec_info import SpecInput, SpecInputType\n",
    "fib import",
)

# (a) capture: EAGLE topk=1 verify wrappers without custom_mask_buf, tagged for the call_begin_forward branch.
t = replace_once(
    t,
    """        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
""",
    """        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            # p6 (SGLANG_EAGLE_SYNCFREE_SEAM): topk=1 chain -> capture causal, no mask buffer.
            p6_chain = _p6_seam.is_eagle_chain_verify(self, forward_mode, spec_info)
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
                and not p6_chain
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
            if p6_chain:
                for w in prefill_wrappers:
                    w._p6_eagle_chain = True
                    w._p6_chain_n = spec_info.draft_token_num
""",
    "capture verify wrappers",
)

# (b) install the sync-free fast_prefill_plan for the tagged verify wrappers (after the capture-time real plan()).
t = replace_once(
    t,
    """            for w in self.prefill_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
""",
    """            for w in self.prefill_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        if (
            in_capture
            and forward_mode.is_target_verify()
            and self.prefill_cuda_graph_metadata.get(bs)
            and all(
                getattr(w, "_p6_eagle_chain", False)
                for w in self.prefill_cuda_graph_metadata[bs]
            )
        ):
            # p6: EAGLE topk=1 verify replays are shape-static per bs like DFLASH;
            # call_begin_forward supplies the host layout (kv_lens = seq_lens_cpu + N).
            for w in self.prefill_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)
            logger.info(
                "[p6] EAGLE topk=1 verify graph bs=%d: causal, no custom mask, fast_prefill_plan",
                bs,
            )

        # Refill the SWA write-target buffer from the live out_cache_loc before
""",
    "install fast plan for EAGLE verify",
)

# (c1) call_begin_forward: identify the tagged wrapper once.
t = replace_once(
    t,
    """        bs = len(seq_lens)
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
""",
    """        bs = len(seq_lens)
        p6_chain = getattr(wrapper_paged, "_p6_eagle_chain", False)
        if p6_chain:
            _p6_seam.validate_chain_spec_info(spec_info, wrapper_paged._p6_chain_n)
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
""",
    "call_begin_forward p6_chain",
)

# (c2) EAGLE generate_attn_arg_prefill: drop the (causal-chain) tree mask for tagged wrappers.
t = replace_once(
    t,
    """            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )
""",
    """            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )
                if p6_chain:
                    # Only real replays spend the per-(kind, bs) check budget:
                    # capture-time dummy inputs carry draft_token=None, idle
                    # inputs an empty one.
                    p6_draft = getattr(spec_info, "draft_token", None)
                    if (
                        p6_draft is not None
                        and p6_draft.numel() > 0
                        and _p6_seam.check_due("verify_mask", bs)
                    ):
                        _p6_seam.check_chain_mask(
                            spec_info,
                            seq_lens_cpu,
                            custom_mask,
                            wrapper_paged._p6_chain_n,
                            bs,
                        )
                    # topk=1: the tree mask IS the causal chain; the wrapper was
                    # captured without a mask buffer and runs MaskMode.CAUSAL.
                    custom_mask = None
""",
    "drop chain mask",
)

# (c3) no host mirror on a tagged wrapper: fall back to the plain (syncing) plan(), still causal, no mask.
t = replace_once(
    t,
    """        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
""",
    """        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        p6_begin_forward = wrapper_paged.begin_forward
        if uses_fast_prefill and p6_chain and seq_lens_cpu is None:
            _p6_seam.warn_once(
                "verify_no_host_lens",
                "EAGLE verify replay without seq_lens_cpu: using the plain plan()",
            )
            uses_fast_prefill = False
            p6_begin_forward = partial(
                BatchPrefillWithPagedKVCacheWrapper.plan, wrapper_paged
            )
""",
    "fast plan fallback",
)

# (c4) EAGLE host layout: seq_lens_cpu excludes the verify window, the device kv lens include it.
t = replace_once(
    t,
    """            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            qo_indptr_host = torch.arange(
""",
    """            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            if p6_chain:
                # EAGLE verify: generate_attn_arg_prefill uses seq_lens + draft_token_num.
                seq_lens_cpu_i32 = seq_lens_cpu_i32 + num_tokens_per_req
            qo_indptr_host = torch.arange(
""",
    "eagle host kv lens",
)

# (c5) warm-up host/device layout assert (default on with the seam: first 8 real replays per bs).
t = replace_once(
    t,
    """                max_kv_len=int(seq_lens_cpu_i32.max()),
            )
""",
    """                max_kv_len=int(seq_lens_cpu_i32.max()),
            )
            if p6_chain and _p6_seam.check_due("verify_indptr", bs):
                _p6_seam.check_verify_indptr(
                    qo_indptr, kv_indptr, qo_indptr_host, kv_indptr_host, bs
                )
""",
    "verify indptr check",
)

# (c6) plan call: pinned-buffer fence around seam fast plans; plain-plan fallback.
t = replace_once(
    t,
    """        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            **paged_plan_kwargs,
        )
""",
    """        p6_fence = uses_fast_prefill and _p6_seam.enabled()
        if p6_fence:
            _p6_seam.plan_fence_wait(wrapper_paged)
        p6_begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            **paged_plan_kwargs,
        )
        if p6_fence:
            _p6_seam.plan_fence_record(wrapper_paged)
""",
    "prefill begin_forward",
)

# (e) Opt3: draft-decode kv_indptr from the host on CUDA-graph replays.
t = replace_once(
    t,
    """    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
""",
    """    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
        host_indptr: bool = False,
    ):
""",
    "common_template signature",
)
t = replace_once(
    t,
    """        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
""",
    """        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = None
        if (
            host_indptr
            and self.topk == 1
            and self.page_size == 1
            and _p6_seam.enabled("draft")
        ):
            # p6: same formula as generate_draft_decode_kv_indices, no blocking D2H.
            indptr_cpu_whole = _p6_seam.host_draft_kv_indptr(
                forward_batch, bs, self.speculative_num_steps
            )
            if indptr_cpu_whole is not None and _p6_seam.check_due(
                "draft_indptr", bs
            ):
                _p6_seam.check_draft_kv_indptr(
                    indptr_cpu_whole, self.kv_indptr[:, : bs + 1], forward_batch, bs
                )
        if indptr_cpu_whole is None:
            indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
""",
    "draft indptr",
)
t = replace_once(
    t,
    "        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)\n",
    "        self.common_template(\n"
    "            forward_batch,\n"
    "            self.cuda_graph_kv_indices,\n"
    "            call_fn,\n"
    "            host_indptr=not in_capture,\n"
    "        )\n",
    "draft out_graph call",
)
FIB.write_text(t)

# ------------------------------------------------------------------------------------------ overlap_utils.py
o = OVL.read_text()
o = replace_once(
    o,
    "from sglang.srt.runtime_context import (\n    get_exec,\n    get_spec,\n)\n",
    "from sglang.srt.runtime_context import (\n    get_exec,\n    get_spec,\n)\n"
    "from sglang.srt.speculative import syncfree_seam as _p6_seam\n",
    "ovl import",
)
o = replace_once(
    o,
    """        self._publish_fresh = False

        self.confidence_relay = ConfidenceRelay(
""",
    """        self._publish_fresh = False
        # p6 (SGLANG_EAGLE_SYNCFREE_SEAM, part d2h): early seq_lens D2H queued at
        # publish; armed = the latest publish has its own copy + done-event.
        self._p6_d2h_done = None
        self._p6_d2h_armed = False

        self.confidence_relay = ConfidenceRelay(
""",
    "future_map init",
)
o = replace_once(
    o,
    """        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()
""",
    """        if self._p6_d2h_armed:
            # p6: publish() already queued this D2H right behind the publish event;
            # wait for its done-event (SGLANG_EAGLE_SYNCFREE_SEAM_WAIT event|spin|poll).
            self._p6_d2h_armed = False
            _p6_seam.wait_d2h(self._p6_d2h_done)
        else:
            # Mechanism: don't sync the schedule stream; gate a private stream on the
            # publish event and copy into the static pinned buffer.
            self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
            with torch.get_device_module(self.device).stream(
                self.fwd_prepare_d2h_stream
            ):
                self.new_seq_lens_cpu_pinned.copy_(
                    self.new_seq_lens_buf, non_blocking=True
                )
            self.fwd_prepare_d2h_stream.synchronize()
""",
    "resolve wait",
)
o = replace_once(
    o,
    """    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
""",
    """    def _p6_issue_early_seq_lens_d2h(self) -> None:
        \"\"\"p6: queue the new_seq_lens D2H right behind publish_ready and record a
        done-event; resolve_seq_lens_cpu() then waits on it instead of issuing its
        own copy behind copy_result's D2H.\"\"\"
        stream = self.fwd_prepare_d2h_stream
        stream.wait_event(self.publish_ready)
        if self._p6_d2h_done is None:
            self._p6_d2h_done = _p6_seam.new_d2h_done_event(self.device)
        with torch.get_device_module(self.device).stream(stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self._p6_d2h_done.record(stream)
        self._p6_d2h_armed = True

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
""",
    "early d2h method",
)
o = replace_once(
    o,
    """            self.publish_ready.record()
            self._publish_fresh = True
""",
    """            self.publish_ready.record()
            self._publish_fresh = True
            if (
                _is_cuda
                and not _DEBUG_ASSERT
                and self.needs_cpu_seq_lens
                and self.fwd_prepare_d2h_stream is not None
                and _p6_seam.enabled("d2h")
            ):
                self._p6_issue_early_seq_lens_d2h()
""",
    "publish early d2h",
)
OVL.write_text(o)
print("p6_syncfree_seam applied")
