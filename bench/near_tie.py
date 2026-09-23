#!/usr/bin/env python3
"""Where does the server's greedy text leave the stored ab_bench reference, and is that token a near-tie?

Runs ab_bench's fixed correctness prompt through raw /generate at temperature 0 and asks for the top-2 logprobs
of the OUTPUT tokens only (the prompt is about 20 tokens, so no large logit buffers). A divergence whose top-1 vs
top-2 margin is below ~0.1 nats is a near-tie (numerically benign); a larger margin means the change moved the
model's numbers by more than rounding noise. Run it on both arms to see the reference margin at the same token.
Env: SGL_ROOT (default /opt/sglang), SGL_BASE (default http://127.0.0.1:30000), SGL_MODEL (default qwen3.8-27b).
"""
import json, os, urllib.request

ROOT = os.environ.get("SGL_ROOT", "/opt/sglang")
BASE = os.environ.get("SGL_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("SGL_MODEL", "qwen3.8-27b")
KEY = open(os.path.join(ROOT, "api_key")).read().strip()
P = "Explain step by step why the sky is blue, then list three related phenomena."
ref = json.load(open(os.path.join(ROOT, "ab", f"reference_{MODEL}.json")))["text"]
body = {"text": P, "sampling_params": {"max_new_tokens": 160, "temperature": 0},
        "return_logprob": True, "top_logprobs_num": 2, "return_text_in_logprobs": True}
r = urllib.request.Request(BASE + "/generate", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
d = json.load(urllib.request.urlopen(r, timeout=300))
cur = d["text"]
k = 0
while k < min(len(cur), len(ref)) and cur[k] == ref[k]:
    k += 1
print(f"common prefix {k} chars of {len(ref)} (reference) / {len(cur)} (now)")
mi = d["meta_info"]
toks = mi.get("output_token_logprobs") or []
tops = mi.get("output_top_logprobs") or []
pos = 0
for i, t in enumerate(toks):
    txt = t[2] if len(t) > 2 and t[2] is not None else ""
    if pos + len(txt) > k:
        top = tops[i] if i < len(tops) else []
        print(f"first differing token #{i}: chosen {txt!r} logprob {t[0]:.4f}")
        for j, e in enumerate(top[:2]):
            print(f"   top{j + 1}: {e[2]!r} logprob {e[0]:.4f}")
        if len(top) >= 2:
            margin = top[0][0] - top[1][0]
            print(f"margin top1-top2 = {margin:.4f} nats -> {'NEAR-TIE' if margin < 0.1 else 'NOT a near-tie'}")
        break
    pos += len(txt)
else:
    print("no divergence from the reference in the returned tokens")
