#!/usr/bin/env python3
"""k1 p3 (plan rank 3): FlashInfer-autotune the target lm_head NVFP4 GEMM, optionally pin its tactic.

Why: the target decode/verify autotune runs with run_lm_head=False (base_runner.py:331), so the verify lm_head
(CutlassFp4GemmRunner 5120 -> 248320, 715 MB) is captured with FlashInfer's fallback tactic -1 (128x128x256, no
swap-AB, 2 stages): 591 us at 1.21 TB/s instead of ~395-420 us with a tuned 128x32x256 tile.

Env (read at call time; all unset = today's code path, same cache file, same tactics, same graphs):
  SGLANG_AUTOTUNE_TARGET_LMHEAD=1
      The target (non-draft) runner's decode/verify autotune dummy forward runs with run_lm_head=True, so the lm_head
      GEMM gets tuned entries for the verify buckets (M=4..20 -> keys 1,2,4,8,16,32) before the verify graphs are
      captured. The verify buffers already carry a logits buffer (base_runner.py:106-114); the draft autotune has
      always run this way (decode_cuda_graph_runner.py:1218, run_lm_head=True).
      Cache key/location: FlashInfer's cache key (flashinfer_autotune_cache_path) does not include run_lm_head, so
      with the flag the target uses its OWN file: key parts + "target_lmhead=1" -> a new
      <SGLANG_CACHE_DIR>/flashinfer/autotune/<ver>/sm120/<new16hex>/rank_tp0_pp0_dp0.json. On first use it is seeded
      by copying the existing target file (764010862dbba616 in production), so only the lm_head shape is profiled
      (a few seconds) and every other target tactic is byte-identical to today. The base target file is never
      written while the flag is on, so unsetting the flag restores today's behaviour exactly (a shared file would
      keep the tuned lm_head entry after the flag is turned off).
      Draft isolation: FlashInfer searches its in-memory profiling cache BEFORE the loaded file, and that cache is
      process-global, so lm_head tactics profiled by the target pass would shadow the draft file's own lm_head
      entries (3f399e09480715e6: M1->6, 2->14, 4->6, 8->6) and could be written into the draft file on its next
      save. After the target pass, isolate_target_lmhead_tactics() moves exactly the lm_head entries this pass
      profiled from the in-memory cache into FlashInfer's loaded-file table: the target verify graphs (captured
      next) still hit them, and the draft's autotune(cache=...) entry clears that table and loads its own file, so
      the draft lm_head tactics, the draft cache file and the eager (runtime) lm_head lookups stay as today.
  SGLANG_LMHEAD_FORCE_TACTIC=<int>   (optional, independent of the flag above)
      Pin the lm_head (N = lm_head out features, M <= SGLANG_LMHEAD_FORCE_TACTIC_MAX_M, default 32) to one
      CutlassFp4GemmRunner tactic at lookup time (not during tuning). This pins target verify AND the shared draft
      lm_head. Tactic ids on SM120: 4 * tile + variant, variant 0 swapAB-DP, 1 noswap-DP, 2 swapAB-StreamK,
      3 noswap-StreamK; tile 1 = 128x32x256 (fp4_gemm_cutlass_template_sm120.h getConfigs). So 4 = 128x32x256
      swap-AB data-parallel (no split-K reduction), 6 = the same tile with StreamK.
      Determinism: on SM120 cutlass::gemm::StreamKScheduler maps to PersistentTileSchedulerSm100StreamK
      (cutlass tile_scheduler.hpp:387-398), which takes the Sm90 stream-K Arguments whose reduction_mode defaults to
      ReductionMode::Deterministic (sm90_tile_scheduler_stream_k.hpp:200); FlashInfer's prepareGemmArgs
      (fp4_gemm_template_sm120.h:138-143) only sets max_swizzle_size and raster_order. So StreamK partial tiles are
      reduced in a fixed turnstile order and a given shape is run-to-run reproducible (tests/t_gpu_k1.py checks it
      bitwise on the GPU). Use SGLANG_LMHEAD_FORCE_TACTIC=4 only if the temp-0 3-run test says otherwise (DP,
      ~2-3% slower). It also serves as the no-autotune fallback.

Applied at image build time. Every edit anchors on exact v0.5.20 text and asserts a single match, so a changed
upstream file fails the build instead of producing a half-patched server.
"""
import pathlib

ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang")
FA = ROOT / "srt/model_executor/runner/flashinfer_autotune.py"
BR = ROOT / "srt/model_executor/runner/base_runner.py"


def replace_once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"p3 {what}: expected 1 anchor, found {n}"
    return text.replace(old, new, 1)


# ---------------- flashinfer_autotune.py ----------------
t = FA.read_text()
assert "target_lmhead_autotune_enabled" not in t, "p3 already applied (flashinfer_autotune)"

t = replace_once(
    t,
    "import json\nimport logging\nfrom pathlib import Path\n",
    "import json\nimport logging\nimport os\nimport shutil\nfrom pathlib import Path\n",
    "imports",
)

t = replace_once(
    t,
    "FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()\n",
    '''FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()

# --- k1 p3: target lm_head autotune + optional lm_head tactic pin (env-gated, default off) ---
_K1_TARGET_LMHEAD_ENV = "SGLANG_AUTOTUNE_TARGET_LMHEAD"
_K1_LMHEAD_TACTIC_ENV = "SGLANG_LMHEAD_FORCE_TACTIC"
_K1_LMHEAD_TACTIC_MAX_M_ENV = "SGLANG_LMHEAD_FORCE_TACTIC_MAX_M"
_K1_LMHEAD_PIN_INSTALLED = False


def _k1_env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def target_lmhead_autotune_enabled(model_runner: ModelRunner) -> bool:
    """SGLANG_AUTOTUNE_TARGET_LMHEAD=1 on a target (non-draft) runner."""
    return _k1_env_flag(_K1_TARGET_LMHEAD_ENV) and not getattr(
        model_runner, "is_draft_worker", False
    )


def _k1_target_lmhead_cache_path(
    model_runner: ModelRunner, model_key_parts: list, flashinfer_version: str, arch: str
) -> Path:
    """Own cache file for the lm_head-tuned target, seeded from the base target file."""
    mr = model_runner
    file_name = f"rank_tp{mr.ps.tp_rank}_pp{mr.ps.pp_rank}_dp{mr.ps.dp_rank or 0}.json"
    root = Path(envs.SGLANG_CACHE_DIR.get()) / "flashinfer" / "autotune" / flashinfer_version / arch
    base_key = hashlib.sha256("|".join(model_key_parts).encode()).hexdigest()[:16]
    key = hashlib.sha256(
        "|".join(list(model_key_parts) + ["target_lmhead=1"]).encode()
    ).hexdigest()[:16]
    cache_dir = root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / file_name
    base_file = root / base_key / file_name
    if not cache_file.exists() and base_file.is_file():
        try:
            shutil.copyfile(base_file, cache_file)
            logger.info(
                "FlashInfer autotune (target lm_head): seeded %s from %s", cache_file, base_file
            )
        except OSError as e:
            logger.warning("FlashInfer autotune (target lm_head): seeding failed: %s", e)
    return cache_file


def _k1_lm_head_out_features(model_runner: ModelRunner) -> Optional[int]:
    model = getattr(model_runner, "model", None)
    for owner in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        lm_head = getattr(owner, "lm_head", None) if owner is not None else None
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            return int(weight.shape[0])
    return None


def _k1_key_bucket(key):
    try:
        return int(key.nearest_profile[0][0])
    except (TypeError, IndexError, ValueError, AttributeError):
        return -1


def _k1_is_lmhead_fp4_key(key, n: int) -> bool:
    if getattr(key, "custom_op", None) != "fp4_gemm":
        return False
    profile = getattr(key, "nearest_profile", None)
    try:
        return len(profile) > 1 and int(profile[1][-1]) == n
    except (TypeError, IndexError, ValueError):
        return False


def snapshot_autotune_keys() -> Optional[set]:
    """Keys in FlashInfer's in-memory profiling cache (None if unavailable)."""
    try:
        from flashinfer.autotuner import AutoTuner

        tuner = AutoTuner.get()
        with tuner._lock:
            return set(tuner.profiling_cache.keys())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("FlashInfer autotune (target lm_head): no key snapshot: %s", e)
        return None


def isolate_target_lmhead_tactics(model_runner: ModelRunner, keys_before) -> None:
    """Move the lm_head tactics the target pass just profiled out of the process-global in-memory cache
    (searched first, also by the draft) into the loaded-file table, which the target verify capture still
    searches and the draft's autotune(cache=...) entry clears before loading the draft file."""
    if keys_before is None:
        return
    n = _k1_lm_head_out_features(model_runner)
    if n is None:
        logger.warning("FlashInfer autotune (target lm_head): lm_head not found, nothing isolated")
        return
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    moved = []
    with tuner._lock:
        for key in list(tuner.profiling_cache.keys()):
            if key in keys_before or not _k1_is_lmhead_fp4_key(key, n):
                continue
            tactic, _ = tuner.profiling_cache.pop(key)
            tuner._file_configs[key.file_key] = (key.runner_class_name, tactic)
            moved.append((_k1_key_bucket(key), key.runner_class_name, tactic))
    log_info_on_rank0(
        logger,
        f"FlashInfer autotune (target lm_head N={n}): {len(moved)} tuned entries kept target-only "
        f"(bucket, runner, tactic): {sorted(moved, key=str)}",
    )


def maybe_install_lmhead_tactic_pin(model_runner: ModelRunner) -> None:
    """SGLANG_LMHEAD_FORCE_TACTIC=<t>: pin the lm_head CutlassFp4GemmRunner tactic at lookup time."""
    global _K1_LMHEAD_PIN_INSTALLED
    raw = os.environ.get(_K1_LMHEAD_TACTIC_ENV, "").strip()
    if not raw or _K1_LMHEAD_PIN_INSTALLED or getattr(model_runner, "is_draft_worker", False):
        return
    try:
        forced = int(raw)
        max_m = int(os.environ.get(_K1_LMHEAD_TACTIC_MAX_M_ENV, "32"))
    except ValueError:
        logger.warning("%s / %s must be ints; no lm_head tactic pin", _K1_LMHEAD_TACTIC_ENV, _K1_LMHEAD_TACTIC_MAX_M_ENV)
        return
    n = _k1_lm_head_out_features(model_runner)
    if n is None:
        logger.warning("%s set but the lm_head was not found; no pin", _K1_LMHEAD_TACTIC_ENV)
        return
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    original_choose_one = tuner.choose_one
    seen = set()

    def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = original_choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        if custom_op != "fp4_gemm" or tuner.is_tuning_mode:
            return runner, tactic
        try:
            m = int(inputs[0].shape[0])
            if int(inputs[1].shape[-1]) != n or m > max_m:
                return runner, tactic
        except Exception:
            return runner, tactic
        for r in runners:
            if type(r).__name__ != "CutlassFp4GemmRunner":
                continue
            if forced not in r.get_valid_tactics(inputs, None):
                if "invalid" not in seen:
                    seen.add("invalid")
                    logger.warning("%s=%d is not a valid CutlassFp4GemmRunner tactic; not pinned", _K1_LMHEAD_TACTIC_ENV, forced)
                return runner, tactic
            if m not in seen:
                seen.add(m)
                logger.info(
                    "lm_head fp4_gemm N=%d M=%d: tactic pinned %s -> %d (%s)",
                    n, m, tactic, forced, _K1_LMHEAD_TACTIC_ENV,
                )
            return r, forced
        return runner, tactic

    tuner.choose_one = choose_one
    _K1_LMHEAD_PIN_INSTALLED = True
    logger.info("lm_head tactic pin installed: N=%d M<=%d tactic=%d", n, max_m, forced)
''',
    "helpers",
)

t = replace_once(
    t,
    "    if mr.is_draft_worker:\n"
    "        model_key_parts.append(f\"draft_quant={mr.model_config.quantization}\")\n"
    "    model_key = \"|\".join(model_key_parts)\n",
    "    if mr.is_draft_worker:\n"
    "        model_key_parts.append(f\"draft_quant={mr.model_config.quantization}\")\n"
    "    if target_lmhead_autotune_enabled(mr):\n"
    "        # k1 p3: separate, seeded cache file; the base target file stays untouched.\n"
    "        return _k1_target_lmhead_cache_path(\n"
    "            mr, model_key_parts, flashinfer_version, arch\n"
    "        )\n"
    "    model_key = \"|\".join(model_key_parts)\n",
    "cache path branch",
)
FA.write_text(t)

# ---------------- base_runner.py ----------------
b = BR.read_text()
assert "isolate_target_lmhead_tactics" not in b, "p3 already applied (base_runner)"
b = replace_once(
    b,
    "from sglang.srt.model_executor.runner.flashinfer_autotune import (\n"
    "    maybe_flashinfer_autotune_extend,\n"
    "    run_flashinfer_autotune_forward,\n"
    "    should_run_flashinfer_autotune,\n"
    ")\n",
    "from sglang.srt.model_executor.runner.flashinfer_autotune import (\n"
    "    isolate_target_lmhead_tactics,\n"
    "    maybe_flashinfer_autotune_extend,\n"
    "    maybe_install_lmhead_tactic_pin,\n"
    "    run_flashinfer_autotune_forward,\n"
    "    should_run_flashinfer_autotune,\n"
    "    snapshot_autotune_keys,\n"
    "    target_lmhead_autotune_enabled,\n"
    ")\n",
    "imports",
)
b = replace_once(
    b,
    "        if mr.device != \"cuda\":\n"
    "            return\n"
    "\n"
    "        self._pre_initialize_flashinfer_allreduce_workspace()\n",
    "        if mr.device != \"cuda\":\n"
    "            return\n"
    "\n"
    "        # k1 p3: no-op unless SGLANG_LMHEAD_FORCE_TACTIC is set (target runner only).\n"
    "        maybe_install_lmhead_tactic_pin(mr)\n"
    "\n"
    "        self._pre_initialize_flashinfer_allreduce_workspace()\n",
    "warmup pin",
)
b = replace_once(
    b,
    "        run_flashinfer_autotune_forward(\n"
    "            self.model_runner, forward_fn, run_lm_head=False\n"
    "        )\n",
    "        # k1 p3 (env SGLANG_AUTOTUNE_TARGET_LMHEAD=1, default off): also tune the target\n"
    "        # lm_head so the verify graphs capture a tuned tactic instead of fallback -1.\n"
    "        tune_lm_head = target_lmhead_autotune_enabled(self.model_runner)\n"
    "        keys_before = snapshot_autotune_keys() if tune_lm_head else None\n"
    "        run_flashinfer_autotune_forward(\n"
    "            self.model_runner, forward_fn, run_lm_head=tune_lm_head\n"
    "        )\n"
    "        if tune_lm_head:\n"
    "            isolate_target_lmhead_tactics(self.model_runner, keys_before)\n",
    "run_lm_head",
)
BR.write_text(b)
print("p3_lmhead_autotune applied")
