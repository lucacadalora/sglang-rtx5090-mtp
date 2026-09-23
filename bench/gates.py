#!/usr/bin/env python3
"""Functional quality gates for a model or flag change on an SGLang server (run inside the sglang image, host network).

  template   /v1/tokenize (no GPU work) on fixed agent-style conversations; prints a sha256 per case. Run it on
             the old and the new server: the hashes must be identical when template + tokenizer are unchanged.
             Also checks that the official template's 400 errors still happen.
  quick      auth, thinking split, tool calls (typed args), image input, short decode speed
  long N     recall test at ~N prompt tokens (6 codes + 2 two-hop facts), decode speed at that depth (a 700-word
             reply sampled at temperature 0.6), then the churn test: 3 distinct background prompts, then the long
             conversation again -> cached_tokens
  churn N K  how many distinct background requests a long conversation survives in the prefix cache

Usage (as root in WSL; the image keeps its Python env in /opt/sglang, so mount the host folder elsewhere). The
tokenizer path below points into the HF cache, so the long test needs the cache mounted too:
  docker run --rm --network host -e SGL_ROOT=/srv/sglang -v /opt/sglang:/srv/sglang:ro -v "$PWD/bench:/bench:ro" \
    --entrypoint python3 lmsysorg/sglang:v0.5.20-cu130 /bench/gates.py quick
"""
import base64, hashlib, json, os, struct, sys, time, urllib.error, urllib.request, zlib

BASE = os.environ.get("SGL_BASE", "http://127.0.0.1:30000")
ROOT = os.environ.get("SGL_ROOT", "/opt/sglang")
KEY = open(f"{ROOT}/api_key").read().strip()
MODEL = os.environ.get("SGL_MODEL") or "qwen3.8-27b"
TOKENIZER = os.environ.get("SGL_TOKENIZER") or f"{ROOT}/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33/tokenizer.json"
# Chat-template kwargs an agent client sends with thinking on and reasoning effort 'low'
BOT_KWARGS = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"}


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + KEY)
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, max_tokens=256, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(extra)
    t0 = time.perf_counter()
    t_first, usage, text, reasoning, tool_deltas, finish = None, None, [], [], [], None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                finish = c.get("finish_reason") or finish
                if (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")) and t_first is None:
                    t_first = time.perf_counter()
                text.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                if delta.get("tool_calls"):
                    tool_deltas.extend(delta["tool_calls"])
    t1 = time.perf_counter()
    ct = (usage or {}).get("completion_tokens", 0)
    return {"ttft_s": round((t_first or t1) - t0, 2), "total_s": round(t1 - t0, 2),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": ct,
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "decode_tok_s": round((ct - 1) / (t1 - t_first), 1) if t_first and ct > 1 and t1 > t_first else None,
            "finish": finish, "text": "".join(text), "reasoning": "".join(reasoning), "tool_deltas": tool_deltas}


TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "set_reminder", "description": "Create a reminder.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "minutes": {"type": "integer", "description": "Minutes from now"},
            "urgent": {"type": "boolean"}}, "required": ["text", "minutes", "urgent"]}}},
]

TEMPLATE_CASES = {
    "system_tools_user": ([{"role": "system", "content": "You are a helpful assistant for a shop owner in Jakarta."},
                           {"role": "user", "content": "Weather in Jakarta, and remind me in 30 minutes to call Budi (urgent)."}], True, BOT_KWARGS),
    "multi_turn_tools": ([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Jakarta, and remind me in 30 minutes to call Budi (urgent)."},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "Two tools: weather and reminder.",
         "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Jakarta", "unit": "celsius"})}},
             {"id": "c2", "type": "function", "function": {"name": "set_reminder", "arguments": json.dumps({"text": "call Budi", "minutes": 30, "urgent": True})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 31, \"sky\": \"humid\"}"},
        {"role": "tool", "tool_call_id": "c2", "content": "{\"ok\": true, \"id\": 17}"},
        {"role": "assistant", "content": "It is 31 C and humid; reminder set.", "reasoning_content": ""},
        {"role": "user", "content": "Thanks. Anything else I should know?"}], True, BOT_KWARGS),
    "multi_turn_nothink": ([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
        {"role": "user", "content": "And times 3?"}], False, {"enable_thinking": False}),
    "image_user": ([{"role": "user", "content": [{"type": "text", "text": "Describe this."},
                                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,PLACEHOLDER"}}]}], False, BOT_KWARGS),
}


def tokenize(messages, tools, kwargs):
    body = {"model": MODEL, "messages": messages, "chat_template_kwargs": kwargs}
    if tools:
        body["tools"] = TOOLS
    with post("/v1/tokenize", body, timeout=60) as r:
        return json.load(r)["tokens"]


def run_template():
    out = {}
    for name, (msgs, tools, kw) in TEMPLATE_CASES.items():
        if name == "image_user":
            msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe this."},
                                                 {"type": "image_url", "image_url": {"url": red_png(16)}}]}]
        try:
            toks = tokenize(msgs, tools, kw)
            out[name] = [len(toks), hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]]
        except urllib.error.HTTPError as e:
            out[name] = ["HTTP", e.code]
    for name, msgs, kw in [("err_effort_high", [{"role": "user", "content": "hi"}], {"reasoning_effort": "high"}),
                           ("err_developer_role", [{"role": "developer", "content": "x"}, {"role": "user", "content": "hi"}], BOT_KWARGS)]:
        try:
            tokenize(msgs, False, kw)
            out[name] = "accepted"
        except urllib.error.HTTPError as e:
            out[name] = f"HTTP {e.code}"
    print(json.dumps(out, sort_keys=True))


def red_png(size=64):
    raw = b"".join(b"\x00" + b"\xdc\x14\x3c" * size for _ in range(size))  # crimson rows
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def run_quick():
    ok = True
    # A chat request without a key must be refused.
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode())
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=30)
        print("AUTH: FAIL (request without key accepted)"); ok = False
    except urllib.error.HTTPError as e:
        print("AUTH: no key ->", e.code)

    r = chat([{"role": "user", "content": "What is 17*23? Answer with the number only."}], 4096, chat_template_kwargs=BOT_KWARGS)
    good = "391" in r["text"] and len(r["reasoning"]) > 0 and "<think>" not in r["text"]
    ok &= good
    print(f"THINKING on: {'PASS' if good else 'FAIL'} answer={r['text'].strip()[:40]!r} reasoning_chars={len(r['reasoning'])} finish={r['finish']}")
    r = chat([{"role": "user", "content": "What is 17*23? Answer with the number only."}], 256, chat_template_kwargs={"enable_thinking": False})
    good = "391" in r["text"] and not r["reasoning"]
    ok &= good
    print(f"THINKING off: {'PASS' if good else 'FAIL'} answer={r['text'].strip()[:40]!r} reasoning_chars={len(r['reasoning'])}")

    passes = 0
    for i in range(3):
        r = chat([{"role": "user", "content": "Remind me in 45 minutes to pay the Tokopedia invoice. It is urgent. Use the tool."}],
                 4096, tools=TOOLS, chat_template_kwargs=BOT_KWARGS)
        names = [t.get("function", {}).get("name") for t in r["tool_deltas"] if t.get("function", {}).get("name")]
        args = "".join(t.get("function", {}).get("arguments") or "" for t in r["tool_deltas"])
        try:
            parsed = json.loads(args) if args else {}
        except json.JSONDecodeError:
            parsed = {}
        good = (names == ["set_reminder"] and parsed.get("minutes") == 45 and parsed.get("urgent") is True
                and r["finish"] == "tool_calls" and "<tool_call>" not in r["text"])
        passes += good
        print(f"TOOL {i+1}: {'PASS' if good else 'FAIL'} names={names} args={args[:90]!r} finish={r['finish']}")
    ok &= passes == 3

    r = chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": red_png()}},
        {"type": "text", "text": "What single colour fills this image? One word."}]}], 2048, chat_template_kwargs=BOT_KWARGS)
    good = any(w in r["text"].lower() for w in ("red", "crimson"))
    ok &= good
    print(f"IMAGE: {'PASS' if good else 'FAIL'} answer={r['text'].strip()[:40]!r}")

    for i in range(2):
        r = chat([{"role": "user", "content": "Explain how a radix tree works and why it suits KV-cache prefix sharing. About 500 words."}],
                 700, chat_template_kwargs={"enable_thinking": False})
        print(f"DECODE short {i+1}: {r['decode_tok_s']} tok/s ({r['completion_tokens']} tokens, ttft {r['ttft_s']} s)")
    print("QUICK:", "PASS" if ok else "FAIL")


CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Palembang", "Denpasar", "Malang", "Padang"]
ITEMS = ["rice", "coffee", "batik", "tea", "cocoa", "nickel", "rubber", "palm oil", "cloves", "pepper", "tin", "copra"]


def filler(n, word="warehouse", seed=0):
    return [f"Day {i}: the {word} in {CITIES[(i + seed) % 10]} shipped {(i * 37 + seed) % 900 + 100} crates of "
            f"{ITEMS[(i + seed) % 12]} to {CITIES[(i * 3 + 1 + seed) % 10]}, and the manifest was signed by clerk "
            f"{(i * 7919 + seed) % 10007}." for i in range(n)]


def build_text(target_tokens, word="warehouse", seed=0, needles=()):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    probe = filler(200, word, seed)
    per = sum(len(e.ids) for e in tok.encode_batch(probe)) / len(probe)
    sents = filler(int(target_tokens / (per + 0.2)), word, seed)
    for frac, text in sorted(needles, reverse=True):
        sents.insert(int(len(sents) * frac), text)
    text = " ".join(sents)
    return text, len(tok.encode(text).ids)


NEEDLES = [
    (0.05, "Note for the auditor: the access code for vault ALPHA is 7Q4-KM2."),
    (0.20, "Note for the auditor: the access code for vault BRAVO is 3XR-9LP."),
    (0.40, "Note for the auditor: the access code for vault CHARLIE is H8D-2WN."),
    (0.55, "Project Heron is led by Dr. Sari Wibowo."),
    (0.60, "Note for the auditor: the access code for vault DELTA is 5TJ-QV7."),
    (0.70, "Captain Budi Hartono commands the cargo ship Merpati."),
    (0.80, "Note for the auditor: the access code for vault ECHO is Z2M-6CK."),
    (0.88, "Dr. Sari Wibowo keeps her office in building 14 of the Cikarang campus."),
    (0.93, "The cargo ship Merpati is registered in the port of Bitung."),
    (0.95, "Note for the auditor: the access code for vault FOXTROT is B9N-4HS."),
]
QUESTIONS = ("From the document, list the access codes for vaults ALPHA, BRAVO, CHARLIE, DELTA, ECHO and FOXTROT, "
             "then answer: (1) in which building number is the office of the person who leads Project Heron? "
             "(2) in which port is the ship commanded by Captain Budi Hartono registered? Answer compactly.")
EXPECT = ["7Q4-KM2", "3XR-9LP", "H8D-2WN", "5TJ-QV7", "Z2M-6CK", "B9N-4HS"]


def run_long(target):
    hay, n = build_text(target, needles=NEEDLES)
    print(f"haystack: {n} tokens")
    doc = [{"role": "system", "content": "You are a careful analyst. Answer only from the document."},
           {"role": "user", "content": "DOCUMENT:\n" + hay},
           {"role": "assistant", "content": "I have read the document."}]
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}

    r = chat(doc + [{"role": "user", "content": QUESTIONS}], 400, temperature=0, **nothink)
    found = [c for c in EXPECT if c in r["text"]]
    hop = ("14" in r["text"], "Bitung" in r["text"])
    new = r["prompt_tokens"] - (r["cached_tokens"] or 0)
    print(f"RECALL cold: {len(found)}/6 codes, two-hop {sum(hop)}/2 | prompt {r['prompt_tokens']} cached {r['cached_tokens']} "
          f"ttft {r['ttft_s']} s (~{round(new / max(r['ttft_s'], 0.01))} tok/s prefill)")
    print("   answer:", r["text"].strip().replace("\n", " ")[:260])

    r = chat(doc + [{"role": "user", "content": "Write about 700 words on what this shipping log suggests about regional trade."}],
             900, temperature=0.6, **nothink)
    print(f"DECODE long: {r['decode_tok_s']} tok/s at ~{r['prompt_tokens']} context | cached {r['cached_tokens']} of {r['prompt_tokens']} ttft {r['ttft_s']} s")

    for i, (size, word) in enumerate([(8000, "depot"), (8000, "harbour"), (16000, "market")]):
        bg, _ = build_text(size, word=word, seed=101 + i)
        r = chat([{"role": "system", "content": f"You summarise {word} logs."},
                  {"role": "user", "content": "Summarise this log in one sentence:\n" + bg}], 120, **nothink)
        print(f"BACKGROUND {i+1}: prompt {r['prompt_tokens']} cached {r['cached_tokens']} ttft {r['ttft_s']} s")

    r = chat(doc + [{"role": "user", "content": "Which vault's code starts with the letter H? Reply with the vault name only."}],
             50, temperature=0, **nothink)
    frac = (r["cached_tokens"] or 0) / max(r["prompt_tokens"] or 1, 1)
    print(f"CHURN test: {'PASS' if frac > 0.95 else 'FAIL'} cached {r['cached_tokens']} of {r['prompt_tokens']} ({frac:.1%}) "
          f"ttft {r['ttft_s']} s answer={r['text'].strip()[:30]!r}")


def mamba_gauges():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            lines = r.read().decode().splitlines()
    except Exception:
        return {}
    out = {}
    for l in lines:
        for k in ("mamba_used_tokens", "mamba_evictable_tokens", "mamba_available_tokens", "evictable_tokens", "num_used_tokens"):
            if l.startswith(f"sglang:{k}") or l.startswith(f"sglang_{k}"):
                out[k] = int(float(l.rsplit(" ", 1)[1]))
    return out


def run_churn(target, max_bg=4, bg_tokens=8000):
    """How many distinct background requests the long conversation survives: after k backgrounds, is it still cached?"""
    hay, n = build_text(target, needles=NEEDLES)
    doc = [{"role": "system", "content": "You are a careful analyst. Answer only from the document."},
           {"role": "user", "content": "DOCUMENT:\n" + hay},
           {"role": "assistant", "content": "I have read the document."}]
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    q = 0
    def main_turn():
        nonlocal q
        q += 1
        return chat(doc + [{"role": "user", "content": f"Question {q}: which vault's code starts with the letter H? One word."}],
                    30, temperature=0, **nothink)
    r = main_turn()
    print(f"main warm-up: prompt {r['prompt_tokens']} cached {r['cached_tokens']} ttft {r['ttft_s']} s | {mamba_gauges()}")
    seed = 500
    for k in range(1, max_bg + 1):
        for j in range(k):
            seed += 1
            bg, _ = build_text(bg_tokens, word=["depot", "harbour", "market", "mill", "farm"][seed % 5], seed=seed)
            chat([{"role": "system", "content": f"You summarise logs, run {seed}."},
                  {"role": "user", "content": "Summarise in one sentence:\n" + bg}], 60, **nothink)
        g = mamba_gauges()
        r = main_turn()
        frac = (r["cached_tokens"] or 0) / max(r["prompt_tokens"] or 1, 1)
        print(f"after {k} background request(s): main cached {frac:.1%} ttft {r['ttft_s']} s | before main turn: {g}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "churn":
        run_churn(int(sys.argv[2]) if len(sys.argv) > 2 else 60000, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
        sys.exit(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if mode == "template":
        run_template()
    elif mode == "quick":
        run_quick()
    elif mode == "long":
        run_long(int(sys.argv[2]) if len(sys.argv) > 2 else 125000)
    else:
        sys.exit(__doc__)
