#!/usr/bin/env python3
"""Quality A/B probe: 20 multi-step word problems with known answers, thinking ON with reasoning_effort low (what an
agent client sends), greedy. Prints accuracy plus mean completion tokens. Usage: arith_eval.py <label>.
Env: SGL_BASE (default http://127.0.0.1:30000), SGL_ROOT (for api_key, default /opt/sglang), SGL_MODEL."""
import json, os, re, statistics, sys, urllib.request
BASE = os.environ.get("SGL_BASE", "http://127.0.0.1:30000")
KEY = open(os.path.join(os.environ.get("SGL_ROOT", "/opt/sglang"), "api_key")).read().strip()
MODEL = os.environ.get("SGL_MODEL", "qwen3.8-27b")
label = sys.argv[1] if len(sys.argv) > 1 else "run"
ok, toks, wrong = 0, [], []
for i in range(20):
    a, b, c, d = 13 + 7 * i, 17 + 3 * i, 101 + 11 * i, 29 + 5 * i
    want = a * b + c - d
    q = (f"A warehouse has {a} shelves with {b} boxes each. {c} boxes arrive and {d} are shipped. "
         f"How many boxes are there now? End your answer with the final number.")
    body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 3000, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d_ = json.load(urllib.request.urlopen(r, timeout=600))
    content = d_["choices"][0]["message"].get("content") or ""
    nums = re.findall(r"-?\d[\d,]*", content)
    got = int(nums[-1].replace(",", "")) if nums else None
    ok += (got == want)
    if got != want:
        wrong.append((want, got))
    toks.append(d_["usage"]["completion_tokens"])
print(f"{label}: {ok}/20 correct | mean completion tokens {statistics.mean(toks):.0f} | wrong {wrong[:5]}")
