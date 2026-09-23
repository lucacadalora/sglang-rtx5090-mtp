#!/usr/bin/env python3
"""Capture a torch-profiler trace of N decode steps on a RUNNING server while driving S parallel streams.
No restart needed: SGLang v0.5.20 exposes POST /start_profile (admin key). Kernels inside CUDA-graph replays are
traced too (CUPTI works under WSL2).

  python3 bench/profile/prof_run.py <prefix> <streams> [num_steps]        # root in WSL

The trace lands in $SGL_ROOT/sglang-cache/prof/<prefix>-*.trace.json.gz (container path /root/.cache/sglang/prof,
which scripts/launch.sh mounts). Summarise it with prof_analyze.py. Each profiled step of an MTP server is one
draft+verify cycle. Profiling inflates the cycle by roughly 5%, so use it for the breakdown, not for tok/s.
Env: SGL_ROOT (default /opt/sglang), SGL_BASE (default http://127.0.0.1:30000), SGL_MODEL (default qwen3.8-27b).
"""
import json, os, sys, threading, time, urllib.request

ROOT = os.environ.get("SGL_ROOT", "/opt/sglang")
BASE = os.environ.get("SGL_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("SGL_MODEL", "qwen3.8-27b")
KEY = open(os.path.join(ROOT, "api_key")).read().strip()
ADMIN = open(os.path.join(ROOT, "admin_api_key")).read().strip()
prefix, streams = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
OUT_HOST = os.path.join(ROOT, "sglang-cache", "prof")
os.makedirs(OUT_HOST, exist_ok=True)


def post(path, body, key, timeout=600):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST")
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + key)
    return urllib.request.urlopen(r, timeout=timeout)


def chat(i):
    body = {"model": MODEL, "max_tokens": 700, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": f"Write a detailed technical essay (variant {i}) about how GPUs schedule warps."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    with post("/v1/chat/completions", body, KEY) as r:
        d = json.load(r)
    return d["usage"]["completion_tokens"]


# warm the prompts first so the profiled steps are pure decode (prefill cached)
threads = [threading.Thread(target=chat, args=(i,)) for i in range(streams)]
[t.start() for t in threads]; [t.join() for t in threads]
time.sleep(2)

res = [None] * streams


def run(i):
    res[i] = chat(i)


threads = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
[t.start() for t in threads]
time.sleep(1.5)  # let prefill finish and decode settle
with post("/start_profile", {"output_dir": "/root/.cache/sglang/prof", "num_steps": steps, "activities": ["CPU", "GPU"],
                             "with_stack": False, "record_shapes": False, "profile_prefix": prefix}, ADMIN) as r:
    print("start_profile:", r.status, r.read()[:200])
[t.join() for t in threads]
print("completion tokens per stream:", res)
time.sleep(5)
print(sorted(f for f in os.listdir(OUT_HOST) if f.startswith(prefix)))
