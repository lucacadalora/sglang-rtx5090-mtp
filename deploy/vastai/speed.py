#!/usr/bin/env python3
"""Decode speed probe for deploy/vastai (stdlib only).

For each N in --streams it fires N simultaneous streaming chats (thinking off, ~400-token answers, different
topics) and prints what the client sees: per-user tok/s, total tok/s and time to first token. It also reads the
server's own 'gen throughput' for decode batches with exactly N running requests from the server log, which is the
number to trust: with --sleep-on-idle the GPU ramps up during the first second of each run, so client figures read
10-15% low. With speculative decoding the 'accept' column is the mean accepted length per verify step (tokens).

--workload picks the prompts: prose (explanatory paragraphs, where drafters accept least), code, structured (JSON)
or mixed. Speculative speed depends far more on the workload than on anything else, so compare like with like.

  python3 deploy/vastai/speed.py --key-file /workspace/sglang/api_key --log /workspace/sglang/logs/server.log
  python3 deploy/vastai/speed.py ... --workload code --temperature 0 --streams 1
"""
import argparse, json, os, re, statistics, threading, time, urllib.request

CODE_TASKS = [
    "parses an nginx access log and reports the ten slowest endpoints",
    "implements an LRU cache with a time-to-live per entry",
    "converts a Markdown table into a list of dictionaries",
    "rate-limits function calls with a token bucket",
    "walks a directory tree and finds duplicate files by hash",
    "implements Dijkstra's algorithm on an adjacency list",
    "validates and normalises international phone numbers",
    "merges overlapping time intervals from a CSV file",
    "retries an HTTP request with exponential backoff and jitter",
    "computes a rolling median over a stream of numbers",
    "serialises a binary tree to a string and back",
    "schedules jobs with dependencies using a topological sort",
    "diffs two JSON documents and prints the changed paths",
    "implements a tiny tokenizer and parser for arithmetic expressions",
    "watches a folder and batches new files into zip archives",
    "simulates a bank account ledger with overdraft protection",
]
STRUCTURED_TOPICS = [
    "large rivers of Asia", "programming languages created before 1990", "suspension bridges",
    "space probes that left Earth orbit", "national parks in Africa", "classic board games",
    "volcanoes in Indonesia", "metro systems in Europe", "famous mathematicians", "types of renewable power plants",
    "musical instruments of Southeast Asia", "deep-sea creatures", "early personal computers", "desert plants",
    "major ocean currents", "medieval castles",
]
TOPICS = [
    "how a lithium-ion cell degrades over charge cycles",
    "why coastal cities sink faster than sea level rises",
    "how sourdough starters keep a stable microbial balance",
    "what limits the range of an electric ferry",
    "how noise-cancelling headphones handle sudden loud sounds",
    "why some bridges are built with deliberate flexibility",
    "how a heat pump moves heat uphill",
    "why volcanic soil is so fertile",
    "how container ships plan their loading order",
    "what makes a good rice paddy irrigation system",
    "how bats avoid jamming each other's sonar",
    "why glass is transparent but steel is not",
    "how a city decides where to put a new metro line",
    "what happens inside a battery during a cold winter night",
    "how coral reefs recover after bleaching",
    "why airplanes fly at the altitudes they do",
]
DECODE = re.compile(r"Decode batch, #running-req: (\d+)(?:.*?accept len: ([0-9.]+))?.*?gen throughput \(token/s\): ([0-9.]+)")


def prompt(workload, i):
    if workload == "mixed":
        workload = ("prose", "code", "structured")[i % 3]
    if workload == "code":
        return f"Write a complete, well-commented Python module that {CODE_TASKS[i % len(CODE_TASKS)]}. Code only."
    if workload == "structured":
        return (f"Return only a JSON array of 12 objects about {STRUCTURED_TOPICS[i % len(STRUCTURED_TOPICS)]}, each with "
                "the fields name, country, year, category and a one-sentence summary. No prose.")
    return f"Write four detailed paragraphs about {TOPICS[i % len(TOPICS)]}."


def chat(base, key, model, text, max_tokens, temperature, out, i):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0, first, usage = time.perf_counter(), None, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                for c in d.get("choices") or []:
                    if first is None and (c.get("delta") or {}).get("content"):
                        first = time.perf_counter()
        end = time.perf_counter()
        n = (usage or {}).get("completion_tokens") or 0
        out[i] = {"tokens": n, "ttft": (first or end) - t0, "wall": end - t0,
                  "decode": (n - 1) / (end - first) if first and n > 1 and end > first else None}
    except Exception as e:  # noqa: BLE001 - report and keep going
        out[i] = {"error": repr(e)}


def server_tput(log, offset, running):
    """Median server gen throughput and mean accept length for decode batches with exactly `running` requests."""
    if not log or not os.path.exists(log):
        return None, None
    with open(log, "rb") as f:
        f.seek(offset)
        rows = [(float(m.group(3)), float(m.group(2)) if m.group(2) else None)
                for m in DECODE.finditer(f.read().decode("utf-8", "replace")) if int(m.group(1)) == running]
    rows = [r for r in rows if r[0] > 20]   # the first line after idle covers the sleep, not decode
    acc = [a for _, a in rows if a is not None]
    return (statistics.median([t for t, _ in rows]) if rows else None,
            sum(acc) / len(acc) if acc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default=os.path.join(os.environ.get("SERVE_ROOT", "/workspace/sglang"), "api_key"))
    ap.add_argument("--log", default=None, help="server log, for the server-side throughput column")
    ap.add_argument("--streams", default=None, help="comma list; default 1,4,<max running from the log>")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--workload", choices=("prose", "code", "structured", "mixed"), default="prose")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--json-out", default=None, help="append one JSON line per stream level to this file")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    key = open(a.key_file).read().strip()
    if a.streams:
        levels = [int(x) for x in a.streams.split(",")]
    else:
        mr = None
        if a.log and os.path.exists(a.log):
            m = re.findall(r"max_running_requests=(\d+)", open(a.log, errors="replace").read())
            mr = int(m[-1]) if m else None
        levels = sorted({1, 4, mr or 4})
    print(f"workload {a.workload}, temperature {a.temperature}, {a.max_tokens} tokens per reply")
    print(f"{'streams':>7} {'per user':>10} {'total':>9} {'server':>9} {'accept':>6} {'TTFT max':>9}  errors")
    for n in levels:
        offset = os.path.getsize(a.log) if a.log and os.path.exists(a.log) else 0
        out, threads = [None] * n, []
        t0 = time.perf_counter()
        for i in range(n):
            t = threading.Thread(target=chat, args=(a.base, key, a.model, prompt(a.workload, i), a.max_tokens,
                                                    a.temperature, out, i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        time.sleep(1.0)  # let the last decode-batch log line land
        ok = [r for r in out if r and "error" not in r]
        errs = [r["error"] for r in out if r and "error" in r]
        per = statistics.median([r["decode"] for r in ok if r["decode"]]) if ok else 0.0
        total = sum(r["tokens"] for r in ok) / wall if ok else 0.0
        srv, acc = server_tput(a.log, offset, n)
        ttft = max((r["ttft"] for r in ok), default=0.0)
        print(f"{n:>7} {per:>7.1f} t/s {total:>6.0f} t/s {('%6.0f t/s' % srv) if srv else '      n/a':>9} "
              f"{('%6.2f' % acc) if acc else '   n/a'} {ttft:>7.2f} s  {len(errs)}")
        for e in errs[:3]:
            print(f"         error: {e}")
        if a.json_out:
            with open(a.json_out, "a") as f:
                f.write(json.dumps({"label": a.label, "workload": a.workload, "temperature": a.temperature,
                                    "streams": n, "per_user_client": round(per, 1), "total_client": round(total, 1),
                                    "server": round(srv, 1) if srv else None, "accept_len": round(acc, 2) if acc else None,
                                    "ttft_max": round(ttft, 3), "errors": len(errs)}) + "\n")


if __name__ == "__main__":
    main()
