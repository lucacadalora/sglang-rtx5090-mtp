#!/usr/bin/env python3
"""Temp-0 token-level equality harness for k1 A/B gates (written for p6, the sync-free MTP seam). Stdlib only.

Sends 50 fixed, diverse prompts (English prose, code, math, Indonesian, JSON/tool-ish, and seven synthetic 1-2K
token contexts) to an SGLang OpenAI endpoint (/v1/chat/completions), thinking off, max_tokens 256, greedy
(temperature 0, top_p 1, top_k -1, no penalties), and records or compares the outputs.

Run from WSL (the API key file is root-only):
  sudo python3 tok_equal.py record  --out base.json                  # arm A (e.g. seam off), sequential
  sudo python3 tok_equal.py compare --ref base.json --out seam.json  # arm B: re-run and compare
  sudo python3 tok_equal.py diff base.json seam.json                 # compare two records offline
  ... --concurrency 5                                               # 5 concurrent streams (bs up to 5)
  ... --logprobs                                                    # also record per-token top-2 logprobs, so a
                                                                    # divergence can be shown to be a near-tie
  sudo python3 tok_equal.py list                                     # print the prompt set (ids, sizes)
  ... --xlong                                                       # opt-in: 3 extra prompts at ~32K/64K/120K
                                                                    # tokens (ids 51-53), where split-kv chunking
                                                                    # is active; the default 50-prompt set and its
                                                                    # sha are unchanged without the flag
  ... --seam-log <container>                                        # after the run, count the p6 layout-guard
                                                                    # lines in `docker logs <container>` (only
                                                                    # "[p6]" lines are read; nothing is echoed)

p6 (sync-free MTP seam) arm B: output equality alone does NOT prove the host layout is right (a host kv layout
short by N or stale by 1 is bit-identical in the output unless a split-kv chunk boundary falls in the last tokens).
Run arm B with the seam's default layout guard (SGLANG_EAGLE_SYNCFREE_SEAM_CHECK unset: first 8 replays per kind
and bs) or SGLANG_EAGLE_SYNCFREE_SEAM_CHECK=1, add --xlong, and pass --seam-log: the run fails unless the log shows
"[p6] seam check verify_mask/verify_indptr/draft_indptr passed" lines and no "[p6]" assertion.

Gate (plan rank 6): sequential compare 50/50 equal; 49/50 only with a near-tie proof (--logprobs margin, or
bench/near_tie.py). Concurrent runs batch requests differently from run to run (arrival timing), so compare a
concurrent record only against a concurrent record, and measure the noise floor with two runs of the same arm.
WARM UP FIRST: the first run after a server launch differed on the ten long prompts (lazily tuned prefill GEMM
tactics); once warm, repeated runs were identical (53/53). Record or compare only on a warmed-up server.
--logprobs adds logprob work to every decode step; use it in both arms or in neither.
Prefix cache: a prompt served from the radix cache (hybrid GDN state restore) is not guaranteed to decode bit-equal
to a cold prefill, so compare cold with cold: run each arm right after a relaunch (or after the maintenance-window
/flush_cache that ab_bench already does). cached_tokens is recorded per prompt and shown on any difference.

The API key is read from --key-file (default $SGL_ROOT/api_key, SGL_ROOT defaults to /opt/sglang) and is never
printed or written. The server URL defaults to $SGL_BASE or http://127.0.0.1:30000.
Exit code: 0 all equal, 1 any difference or error, 2 usage/connection problem.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("SGL_BASE", "http://127.0.0.1:30000")
DEFAULT_KEY_FILE = os.path.join(os.environ.get("SGL_ROOT", "/opt/sglang"), "api_key")
MAX_TOKENS = 256
NEAR_TIE_NATS = 0.05


# ------------------------------------------------------------------------------------------------ prompt set
class _Lcg:
    """Deterministic generator for the synthetic long contexts (independent of Python's random module)."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def pick(self, seq):
        return seq[self.next() % len(seq)]

    def rng(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)


def _long_log():
    g = _Lcg(44)
    services = ["auth", "billing", "search", "gateway", "inventory", "notifier"]
    levels = ["INFO"] * 6 + ["WARN"] * 2 + ["ERROR"]
    msgs = {
        "INFO": ["request completed", "cache hit ratio updated", "heartbeat ok", "config reloaded",
                 "connection pool resized", "job scheduled"],
        "WARN": ["slow query detected", "retrying upstream call", "queue depth above threshold",
                 "certificate expires soon"],
        "ERROR": ["timeout talking to database", "null pointer in handler", "disk quota exceeded",
                  "upstream returned 502", "failed to parse payload"],
    }
    lines = []
    for i in range(42):
        lvl = g.pick(levels)
        svc = g.pick(services)
        lines.append(f"2026-09-20T08:{i // 60 + 10:02d}:{i % 60:02d}Z {lvl:5s} [{svc}] {g.pick(msgs[lvl])} "
                     f"(req={g.rng(1000, 9999)}, latency_ms={g.rng(3, 2400)})")
    return "\n".join(lines)


def _long_meeting():
    g = _Lcg(45)
    people = ["Rina", "Marco", "Aiko", "Tomas", "Priya"]
    topics = ["the Q4 roadmap", "the database migration", "on-call rotation", "the pricing page redesign",
              "hiring for the platform team", "the customer escalation from Acme", "the load-test results"]
    verbs = ["thinks", "suggests", "is worried that", "confirms that", "proposes that", "notes that"]
    claims = ["we should move the deadline by two weeks", "the staging cluster needs more memory",
              "we can reuse the existing auth service", "the vendor contract ends in March",
              "the latency regression comes from the new serializer", "we need a rollback plan before Friday",
              "the budget allows one more contractor", "documentation is the main blocker"]
    out = ["Meeting notes, platform sync, 18 September 2026. Attendees: " + ", ".join(people) + "."]
    for t in topics:
        out.append(f"\nTopic: {t}.")
        for _ in range(g.rng(10, 14)):
            out.append(f"{g.pick(people)} {g.pick(verbs)} {g.pick(claims)}.")
        owner = g.pick(people)
        out.append(f"Decision: {owner} will own {t} and report back by {g.pick(['Monday', 'Wednesday', 'Friday'])}.")
    return "\n".join(out)


def _long_csv():
    g = _Lcg(46)
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = ["month,region,product,units,unit_price_usd"]
    for m in range(1, 13):
        for r in regions:
            for _ in range(g.rng(1, 2)):
                rows.append(f"2025-{m:02d},{r},{g.pick(products)},{g.rng(10, 400)},{g.rng(5, 90)}")
    return "\n".join(rows)


def _long_code():
    g = _Lcg(47)
    parts = ['"""Inventory helpers (synthetic test module)."""', "", "from dataclasses import dataclass", "", ""]
    parts += ["@dataclass", "class Item:", "    sku: str", "    qty: int", "    price: float", ""]
    for i in range(18):
        name = f"adjust_{g.pick(['stock', 'price', 'batch', 'order'])}_{i}"
        k = g.rng(2, 9)
        parts += [f"def {name}(items, factor={k}):",
                  f'    """Apply rule {i} to every item and return the changed SKUs."""',
                  "    changed = []",
                  "    for it in items:",
                  f"        if it.qty % {k} == 0:",
                  f"            it.price = round(it.price * (1 + factor / 100), 2)",
                  "            changed.append(it.sku)",
                  "    return changed", "", ""]
    parts += ["def total_value(items):",
              '    """Total stock value."""',
              "    total = 0",
              "    for i in range(1, len(items)):",
              "        total += items[i].qty * items[i].price",
              "    return total", ""]
    return "\n".join(parts)


def _long_spec_id():
    g = _Lcg(48)
    kategori = ["laptop", "ponsel", "monitor", "printer", "router"]
    fitur = ["baterai tahan 12 jam", "layar 14 inci", "garansi resmi 2 tahun", "berat 1,3 kg",
             "mendukung Wi-Fi 6", "memori 16 GB", "penyimpanan SSD 512 GB", "port USB-C", "kamera 50 MP"]
    out = ["Katalog produk Toko Elektronik Maju Jaya, edisi September 2026."]
    for i in range(40):
        k = g.pick(kategori)
        out.append(f"Produk {i + 1}: {k} seri {g.pick(['A', 'B', 'C', 'X'])}{g.rng(100, 999)}, "
                   f"harga Rp {g.rng(8, 250) * 100_000:,}".replace(",", ".")
                   + f", {g.pick(fitur)}, {g.pick(fitur)}, stok {g.rng(0, 40)} unit.")
    return "\n".join(out)


def _long_json_cfg():
    g = _Lcg(49)
    services = []
    for i in range(18):
        services.append({"name": f"svc-{g.pick(['api', 'worker', 'cron', 'web', 'db'])}-{i:02d}",
                         "image": f"registry.local/team/{g.pick(['core', 'edge', 'data'])}:{g.rng(1, 9)}.{g.rng(0, 20)}",
                         "replicas": g.rng(1, 5),
                         "healthcheck": {"enabled": g.rng(0, 3) != 0, "path": "/healthz",
                                         "interval_s": g.pick([5, 10, 30])},
                         "env": {"LOG_LEVEL": g.pick(["info", "debug", "warn"]), "REGION": g.pick(["id-1", "sg-1"])}})
    return json.dumps({"version": 3, "services": services}, indent=1)


def _long_terms():
    g = _Lcg(50)
    clauses = [
        "The Supplier shall deliver the Goods to the address stated in the Order within {d} business days.",
        "Either party may terminate this Agreement by giving the other party {n} days' written notice.",
        "Shipping costs for returned Goods shall be borne by the {who} unless the Goods are defective.",
        "Invoices are payable within {d} days of receipt; late payments accrue interest at {p}% per month.",
        "The Supplier warrants that the Goods are free from material defects for {m} months after delivery.",
        "Neither party is liable for delays caused by events beyond its reasonable control.",
        "All notices must be sent by registered mail or email to the addresses in Schedule {s}.",
        "This Agreement is governed by the laws of the Republic of Indonesia.",
    ]
    out = ["MASTER SUPPLY AGREEMENT (synthetic test document)"]
    for i in range(1, 51):
        c = g.pick(clauses)
        if "notice" in c and "terminate" in c:
            c = c.format(n=60)
        elif "{who}" in c:
            c = c.format(who="Customer")
        else:
            c = c.format(d=g.rng(7, 45), p=g.rng(1, 3), m=g.rng(6, 24), s=g.pick("ABC"))
        out.append(f"{i}. {c}")
    return "\n".join(out)


def build_prompts():
    u = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    P = []

    def add(cat, messages):
        P.append({"id": f"{len(P) + 1:02d}-{cat}", "cat": cat, "messages": messages})

    for t in [
        "Write a vivid paragraph describing a thunderstorm rolling over a coastal town at dusk.",
        "Explain to a 12-year-old why the sky is blue, in about 150 words.",
        "Summarize the main causes of the French Revolution in five bullet points.",
        "Write a short, polite email declining a meeting invitation and proposing two alternative times.",
        "Describe the process of photosynthesis step by step.",
        "Write a 200-word short story about a lighthouse keeper who finds a message in a bottle.",
        "What are the pros and cons of remote work for software teams? Be balanced and concrete.",
        "Rewrite in a formal register: 'hey guys, the server's down again, can someone look at it asap?'",
        "List ten creative names for a coffee shop run by retired astronomers, each with a one-line tagline.",
    ]:
        add("prose", u(t))
    for t in [
        "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring "
        "and type hints.",
        "Write a SQL query that finds the top 3 customers by total order value in 2024 from tables "
        "customers(id, name) and orders(id, customer_id, amount, created_at).",
        "Explain what this Bash one-liner does: find . -name '*.log' -mtime +7 -print0 | xargs -0 gzip -9",
        "Implement a thread-safe LRU cache in Rust with get and put methods.",
        "Find and fix the bugs: function sum(arr){ let s; for (let i=0;i<=arr.length;i++){ s+=arr[i]; } "
        "return s; }",
        "Write a C function that reverses a singly linked list in place and explain its time complexity.",
        "Convert this Python loop into a list comprehension and explain the difference:\nresult = []\n"
        "for x in range(20):\n    if x % 3 == 0:\n        result.append(x * x)",
        "Write a Dockerfile for a minimal Flask app using python:3.12-slim, a non-root user and gunicorn.",
        "Implement binary search in Go and include table-driven tests.",
        "Write a TypeScript interface and a zod schema for a user profile with name, email, optional age "
        "and tags.",
    ]:
        add("code", u(t))
    for t in [
        "A train leaves at 09:40 and travels 234 km at an average speed of 78 km/h. At what time does it "
        "arrive? Show your steps.",
        "Solve for x: 3x^2 - 12x + 9 = 0. Show the work.",
        "What is the sum of all integers from 1 to 1000 that are divisible by 3 or 5?",
        "Prove that the square root of 2 is irrational.",
        "A bag has 5 red, 3 blue and 2 green balls. Two are drawn without replacement. What is the "
        "probability that both have the same color?",
        "Compute the derivative of f(x) = x^3 * ln(x) and find its critical points for x > 0.",
        "If 17% of a number is 51, what is 35% of the same number? Explain.",
        "Explain the Monty Hall problem and compute the winning probability when switching.",
    ]:
        add("math", u(t))
    for t in [
        "Jelaskan secara singkat apa itu fotosintesis dan mengapa penting bagi kehidupan di bumi.",
        "Tuliskan resep sederhana nasi goreng untuk dua orang, lengkap dengan bahan dan langkah-langkahnya.",
        "Apa perbedaan antara kecerdasan buatan, pembelajaran mesin, dan pembelajaran mendalam? Jelaskan "
        "dengan contoh.",
        "Buatkan surat lamaran kerja singkat untuk posisi staf administrasi di sebuah perusahaan logistik "
        "di Surabaya.",
        "Terjemahkan ke bahasa Inggris: 'Besok pagi saya akan berangkat ke Bandung dengan kereta api pukul "
        "tujuh.'",
        "Sebutkan lima tips menjaga kesehatan mental bagi mahasiswa yang sedang menyusun skripsi.",
        "Jelaskan sejarah singkat Sumpah Pemuda 1928 dan maknanya bagi bangsa Indonesia.",
        "Hitung: harga sebuah buku Rp 45.000, mendapat diskon 20%, lalu dikenakan pajak 11%. Berapa harga "
        "akhirnya?",
    ]:
        add("indonesian", u(t))
    json_sys = {"role": "system", "content": "You are a JSON API. Reply with valid JSON only, no prose."}
    tool_sys = {"role": "system", "content": "You can call tools by replying with one JSON object "
                "{\"tool\": <name>, \"args\": {...}} per line. Tools: search(query: string), "
                "calculator(expression: string), get_weather(city: string, unit: 'c'|'f')."}
    add("json", [json_sys, {"role": "user", "content": "Extract name, company, email and phone from: 'Hi, I'm "
                            "Dewi Lestari from PT Nusantara Data, reach me at dewi.l@nusantara-data.example "
                            "or +62 812 5555 0199.'"}])
    add("json", [json_sys, {"role": "user", "content": "Return a JSON array of 5 planets with fields name, "
                            "diameter_km, moons, has_rings."}])
    add("json", [tool_sys, {"role": "user", "content": "What's the weather in Jakarta in Celsius? Call the "
                            "right tool."}])
    add("json", u("Convert this CSV to JSON: id,name,qty\n1,widget,12\n2,gadget,7\n3,doohickey,0"))
    add("json", u("Validate this JSON and list every error: {\"user\": \"alice\", \"age\": 31,, \"roles\": "
                  "[\"admin\" \"dev\"], \"active\": tru}"))
    add("json", u("Write an OpenAPI 3.0 YAML snippet for GET /v1/orders/{id} returning an Order with id, "
                  "status and total."))
    add("json", [tool_sys, {"role": "user", "content": "What is 23.5% of 18,400, and who won the 2018 FIFA "
                            "World Cup? Plan the tool calls in order, then emit them."}])
    add("json", u("Produce a JSON Schema (draft 2020-12) for a blog post with title, body, author {name, "
                  "email}, unique string tags and published_at (date-time)."))
    add("long", u("Here is a service log:\n\n" + _long_log() + "\n\nWhich service has the most ERROR lines, "
                  "and what was its first error message? Answer briefly."))
    add("long", u(_long_meeting() + "\n\nSummarize the decisions and the action items with their owners."))
    add("long", u("Sales data (CSV):\n" + _long_csv() + "\n\nWhich region had the highest total revenue "
                  "(units x unit_price_usd) in Q3 (July-September)? Show the calculation briefly."))
    add("long", u("```python\n" + _long_code() + "```\n\nExplain what this module does and point out one bug."))
    add("long", u(_long_spec_id() + "\n\nBerdasarkan katalog di atas, produk mana saja yang stoknya habis "
                  "(0 unit)? Jawab dalam bahasa Indonesia."))
    add("long", u("Deployment config:\n" + _long_json_cfg() + "\n\nList every service with replicas > 2 "
                  "whose healthcheck is disabled."))
    add("long", u(_long_terms() + "\n\nWhat is the notice period for termination, and who bears the "
                  "shipping costs for returned goods?"))
    assert len(P) == 50, len(P)
    return P


def _xlong_log(n_lines, seed):
    """Synthetic service log for the opt-in --xlong prompts (unique req ids, so the retrieval question has one
    answer). Separate from _long_log so the default prompt set stays byte-identical."""
    g = _Lcg(seed)
    services = ["auth", "billing", "search", "gateway", "inventory", "notifier"]
    levels = ["INFO"] * 6 + ["WARN"] * 2 + ["ERROR"]
    msgs = {
        "INFO": ["request completed", "cache hit ratio updated", "heartbeat ok", "config reloaded",
                 "connection pool resized", "job scheduled"],
        "WARN": ["slow query detected", "retrying upstream call", "queue depth above threshold",
                 "certificate expires soon"],
        "ERROR": ["timeout talking to database", "null pointer in handler", "disk quota exceeded",
                  "upstream returned 502", "failed to parse payload"],
    }
    lines = []
    for i in range(n_lines):
        lvl = g.pick(levels)
        svc = g.pick(services)
        t = 8 * 3600 + 7 * i
        lines.append(f"2026-09-20T{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}Z {lvl:5s} [{svc}] "
                     f"{g.pick(msgs[lvl])} (req=R{i:05d}, latency_ms={g.rng(3, 2400)})")
    return "\n".join(lines)


# (id, log lines, seed): measured 31,880 / 64,153 / 120,147 prompt tokens with the served tokenizer and chat
# template; 120K + 256 output tokens stays inside the 147,456-token window.
XLONG_SPECS = (("51", 700, 61), ("52", 1400, 62), ("53", 2630, 63))


def build_xlong_prompts():
    P = []
    for pid, n_lines, seed in XLONG_SPECS:
        text = (f"Here is a service log with {n_lines} lines:\n\n" + _xlong_log(n_lines, seed)
                + "\n\nFirst, give the service and the latency_ms of the line with req=R00007. Then write an "
                "incident summary of about 200 words covering the ERROR and WARN lines in the last 40 lines of "
                "the log.")
        P.append({"id": f"{pid}-xlong", "cat": "xlong", "messages": [{"role": "user", "content": text}]})
    return P


PROMPTS = build_prompts()
PROMPT_SET_SHA = hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()[:16]
XLONG_PROMPTS = build_xlong_prompts()
XLONG_SET_SHA = hashlib.sha256(json.dumps(PROMPTS + XLONG_PROMPTS, sort_keys=True).encode()).hexdigest()[:16]


def prompt_pool(xlong):
    """(prompts, set sha) for a run: the default 50, or the 50 + the 3 opt-in 32K-120K prompts."""
    return (PROMPTS + XLONG_PROMPTS, XLONG_SET_SHA) if xlong else (PROMPTS, PROMPT_SET_SHA)


# ------------------------------------------------------------------------------------------------ p6 seam log
_SEAM_PASS = re.compile(r"\[p6\] seam check (\w+) passed \((?:bs=(\d+), )?(\d+)/(\d+)\)")
SEAM_KINDS = ("verify_mask", "verify_indptr", "draft_indptr")


def parse_seam_log(text):
    """Summarize the p6 layout-guard lines of a server log. Only lines containing "[p6]" are looked at."""
    s = {"seam_on": 0, "passed": {k: {} for k in SEAM_KINDS}, "failures": 0, "verify_graphs": 0}
    for line in text.splitlines():
        if "[p6]" not in line:
            continue
        if "sync-free MTP seam ON" in line:
            s["seam_on"] += 1
        if "EAGLE topk=1 verify graph" in line:
            s["verify_graphs"] += 1
        if "Error: [p6]" in line or "AssertionError" in line:
            s["failures"] += 1
        m = _SEAM_PASS.search(line)
        if m and m.group(1) in s["passed"]:
            bs = m.group(2) or "-"
            s["passed"][m.group(1)][bs] = max(s["passed"][m.group(1)].get(bs, 0), int(m.group(3)))
    return s


def seam_log_verdict(s):
    """(ok, message) for a parse_seam_log summary: seam on, every kind checked at least once, no failures."""
    if s.get("error"):
        return False, f"seam log unavailable: {s['error']}"
    missing = [k for k in SEAM_KINDS if not s["passed"][k]]
    parts = [f"{k} " + (",".join(f"bs{b}:{n}" for b, n in sorted(s["passed"][k].items())) or "NONE")
             for k in SEAM_KINDS]
    msg = (f"seam ON lines {s['seam_on']}, causal verify graphs {s['verify_graphs']}, "
           f"checks passed (highest logged count per bs): {'; '.join(parts)}; [p6] failures {s['failures']}")
    ok = s["seam_on"] > 0 and not missing and s["failures"] == 0
    return ok, msg


def seam_log_summary(container):
    try:
        r = subprocess.run(["docker", "logs", container], capture_output=True, text=True, timeout=120,
                           errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if r.returncode != 0:
        return {"error": f"docker logs {container} exited {r.returncode}"}
    return parse_seam_log(r.stdout + "\n" + r.stderr)


# ------------------------------------------------------------------------------------------------ client
def read_key(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        sys.exit(f"cannot read API key file {path}: {e.strerror} (run with sudo from WSL)")


def http_json(url, key, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {key}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one(args, key, model, p):
    body = {
        "model": model,
        "messages": p["messages"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 2
    t0 = time.time()
    rec = {"id": p["id"], "cat": p["cat"]}
    try:
        r = http_json(args.url.rstrip("/") + "/v1/chat/completions", key, body)
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        rec.update(text=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                   finish_reason=ch.get("finish_reason"),
                   completion_tokens=(r.get("usage") or {}).get("completion_tokens"),
                   prompt_tokens=(r.get("usage") or {}).get("prompt_tokens"),
                   cached_tokens=((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        lp = (ch.get("logprobs") or {}).get("content")
        if lp:
            rec["tokens"] = [e.get("token") for e in lp]
            rec["top2"] = [[[t.get("token"), t.get("logprob")] for t in (e.get("top_logprobs") or [])[:2]]
                           for e in lp]
    except urllib.error.HTTPError as e:
        rec["error"] = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def server_load(args, key):
    """Running + queued requests from /metrics (None if unavailable)."""
    try:
        req = urllib.request.Request(args.url.rstrip("/") + "/metrics")
        req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=5) as r:
            txt = r.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return None
    n, seen = 0.0, False
    for line in txt.splitlines():
        if line.startswith(("sglang:num_running_reqs", "sglang:num_queue_reqs")):
            try:
                n += float(line.rsplit(" ", 1)[1])
                seen = True
            except ValueError:
                pass
    return n if seen else None


def wait_idle(args, key, timeout=180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        load = server_load(args, key)
        if load is None or load == 0:
            return round(time.time() - t0, 2)
        time.sleep(0.25)
    print(f"  WARNING: server still busy after {timeout:.0f}s, sending anyway", flush=True)
    return round(time.time() - t0, 2)


class LoadSampler:
    """Polls /metrics while one sequential request runs; max_load > 1 means other traffic overlapped it."""

    def __init__(self, args, key):
        import threading

        self.args, self.key = args, key
        self.max_load = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(0.2):
            load = server_load(self.args, self.key)
            if load is not None:
                self.max_load = load if self.max_load is None else max(self.max_load, load)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)


def run_all(args):
    key = read_key(args.key_file)
    try:
        model = args.model or http_json(args.url.rstrip("/") + "/v1/models", key)["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach {args.url}/v1/models: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    pool, set_sha = prompt_pool(args.xlong)
    sel = [p for p in pool if not args.ids or p["id"][:2] in args.ids.split(",")]
    print(f"model={model} prompts={len(sel)} concurrency={args.concurrency} logprobs={args.logprobs} "
          f"max_tokens={MAX_TOKENS} set={set_sha}{' (with --xlong)' if args.xlong else ''}", flush=True)
    t0 = time.time()
    results = [None] * len(sel)

    def show(n, r):
        status = "ERR " + r["error"][:80] if "error" in r else f"{r.get('completion_tokens')} tok"
        load_s = "" if r.get("max_load") in (None, 0, 1) else f"  OVERLAP: {r['max_load']:.0f} reqs on server"
        print(f"  [{n:2d}/{len(sel)}] {r['id']:14s} {r['elapsed_s']:6.2f}s {status}{load_s}", flush=True)

    if args.concurrency == 1 and not args.no_idle_check:
        # Sequential gate runs: start each prompt on an idle server and watch for other traffic (a second
        # request changes the batch and can legitimately change greedy output).
        for n, p in enumerate(sel, 1):
            waited = wait_idle(args, key)
            sampler = LoadSampler(args, key)
            sampler.start()
            r = one(args, key, model, p)
            sampler.stop()
            r["idle_wait_s"] = waited
            r["max_load"] = sampler.max_load
            results[n - 1] = r
            show(n, r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(one, args, key, model, p): i for i, p in enumerate(sel)}
            for n, f in enumerate(concurrent.futures.as_completed(futs), 1):
                i = futs[f]
                results[i] = f.result()
                show(n, results[i])
    wall = time.time() - t0
    toks = sum(r.get("completion_tokens") or 0 for r in results)
    print(f"done in {wall:.1f}s, {toks} completion tokens ({toks / wall:.1f} tok/s aggregate)", flush=True)
    meta = {"url": args.url, "model": model, "concurrency": args.concurrency, "logprobs": args.logprobs,
            "max_tokens": MAX_TOKENS, "prompt_set_sha": set_sha, "label": args.label,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(wall, 2), "completion_tokens": toks}
    if args.xlong:
        meta["xlong"] = True
    if args.seam_log:
        s = seam_log_summary(args.seam_log)
        ok, msg = seam_log_verdict(s)
        print(f"p6 {'OK' if ok else 'FAIL'}: {msg}", flush=True)
        meta["seam_log"] = {"container": args.seam_log, "ok": ok, "summary": msg}
    return {"meta": meta, "results": results}


def save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"saved {path}")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ compare
def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def compare(ref, new):
    ma, mb = ref["meta"], new["meta"]
    if ma.get("prompt_set_sha") != mb.get("prompt_set_sha"):
        print(f"WARNING: prompt sets differ ({ma.get('prompt_set_sha')} vs {mb.get('prompt_set_sha')})")
    if ma.get("concurrency") != mb.get("concurrency"):
        print(f"WARNING: concurrency differs ({ma.get('concurrency')} vs {mb.get('concurrency')}); batching "
              "differences can change greedy outputs")
    print(f"ref: {ma.get('label') or ''} {ma.get('utc')} conc={ma.get('concurrency')} "
          f"| new: {mb.get('label') or ''} {mb.get('utc')} conc={mb.get('concurrency')}")
    by_id = {r["id"]: r for r in new["results"]}
    n_eq = n_text = n_cnt = n_err = 0
    rows = []
    for ra in ref["results"]:
        rb = by_id.get(ra["id"])
        if rb is None:
            rows.append(f"  {ra['id']:14s} MISSING in new")
            n_err += 1
            continue
        if "error" in ra or "error" in rb:
            rows.append(f"  {ra['id']:14s} ERROR ref={ra.get('error', 'ok')[:60]} new={rb.get('error', 'ok')[:60]}")
            n_err += 1
            continue
        ta = ra.get("reasoning", "") + "\x00" + ra["text"]
        tb = rb.get("reasoning", "") + "\x00" + rb["text"]
        same_text = ta == tb
        same_cnt = ra.get("completion_tokens") == rb.get("completion_tokens")
        same_tok = True
        tok_i = -1
        if ra.get("tokens") is not None and rb.get("tokens") is not None:
            tok_i = first_diff(ra["tokens"], rb["tokens"])
            same_tok = tok_i < 0
        n_text += same_text
        n_cnt += same_cnt
        if same_text and same_cnt and same_tok:
            n_eq += 1
            continue
        i = first_diff(ra["text"], rb["text"])
        ctx_a = ra["text"][max(0, i - 30): i + 30].replace("\n", "\\n")
        ctx_b = rb["text"][max(0, i - 30): i + 30].replace("\n", "\\n")
        cache_note = ""
        if ra.get("cached_tokens") != rb.get("cached_tokens"):
            cache_note = f", prompt cache hit {ra.get('cached_tokens')} vs {rb.get('cached_tokens')} tokens"
        if max(ra.get("max_load") or 0, rb.get("max_load") or 0) > 1:
            cache_note += (f", OTHER TRAFFIC overlapped (max load {ra.get('max_load')} vs "
                           f"{rb.get('max_load')})")
        line = (f"  {ra['id']:14s} DIFF first char {i} (len {len(ra['text'])} vs {len(rb['text'])}), "
                f"completion tokens {ra.get('completion_tokens')} vs {rb.get('completion_tokens')}{cache_note}\n"
                f"      ref: ...{ctx_a}...\n      new: ...{ctx_b}...")
        if tok_i >= 0:
            line += f"\n      first differing token #{tok_i}"
            for name, r in (("ref", ra), ("new", rb)):
                t2 = (r.get("top2") or [])
                if tok_i < len(t2) and len(t2[tok_i]) == 2:
                    (t1, l1), (t2b, l2) = t2[tok_i]
                    l1, l2 = float(l1 or 0.0), float(l2 or 0.0)
                    margin = l1 - l2
                    tag = "NEAR-TIE" if margin < NEAR_TIE_NATS else "not a near-tie"
                    line += f"\n      {name} top2 at #{tok_i}: {t1!r} {l1:.4f} / {t2b!r} {l2:.4f} margin {margin:.4f} nats ({tag})"
        rows.append(line)
    total = len(ref["results"])
    for r in rows:
        print(r)
    busy = [ra["id"] for ra in ref["results"]
            if max(ra.get("max_load") or 0, (by_id.get(ra["id"]) or {}).get("max_load") or 0) > 1]
    if busy and ma.get("concurrency") == 1:
        print(f"NOTE: {len(busy)} prompts overlapped other traffic in at least one run: {', '.join(busy)}")
    warm = [ra["id"] for ra in ref["results"]
            if (ra.get("cached_tokens") or 0) != ((by_id.get(ra["id"]) or {}).get("cached_tokens") or 0)]
    if warm:
        print(f"NOTE: prefix-cache hits differ between the runs for {len(warm)} prompts (compare cold with cold)")
    print(f"SUMMARY: {n_eq}/{total} equal (text {n_text}/{total}, completion-token counts {n_cnt}/{total}, "
          f"errors {n_err})")
    return 0 if n_eq == total else 1


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--url", default=DEFAULT_URL)
        s.add_argument("--key-file", default=DEFAULT_KEY_FILE)
        s.add_argument("--model", default=None, help="default: first id from /v1/models")
        s.add_argument("--concurrency", type=int, default=1)
        s.add_argument("--logprobs", action="store_true")
        s.add_argument("--ids", default=None, help="comma list of 2-digit prompt ids to run (default all 50)")
        s.add_argument("--label", default=None)
        s.add_argument("--no-idle-check", action="store_true",
                       help="sequential mode: do not wait for an idle server / sample /metrics for overlap")
        s.add_argument("--out", required=(name == "record"))
        s.add_argument("--xlong", action="store_true",
                       help="add the 3 opt-in prompts at ~32K/64K/120K tokens (ids 51-53)")
        s.add_argument("--seam-log", default=None, metavar="CONTAINER",
                       help="p6: after the run, check the layout-guard lines in `docker logs CONTAINER`")
        if name == "compare":
            s.add_argument("--ref", required=True)
    d = sub.add_parser("diff")
    d.add_argument("ref")
    d.add_argument("new")
    ls = sub.add_parser("list")
    ls.add_argument("--xlong", action="store_true")
    args = ap.parse_args()

    if args.cmd == "list":
        pool, set_sha = prompt_pool(args.xlong)
        for p in pool:
            chars = sum(len(m["content"]) for m in p["messages"])
            print(f"{p['id']:14s} {chars:6d} chars  {p['messages'][-1]['content'][:70]!r}")
        print(f"{len(pool)} prompts, set {set_sha}")
        return 0
    if args.cmd == "diff":
        return compare(load(args.ref), load(args.new))
    rec = run_all(args)
    if args.out:
        save(rec, args.out)
    seam_fail = bool(args.seam_log) and not rec["meta"]["seam_log"]["ok"]
    if args.cmd == "compare":
        return max(compare(load(args.ref), rec), int(seam_fail))
    return 1 if seam_fail or any("error" in r for r in rec["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
