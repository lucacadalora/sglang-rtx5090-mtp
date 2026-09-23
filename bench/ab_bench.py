#!/usr/bin/env python3
"""A/B harness for serving-stack changes. Stdlib only. Run as root in WSL (needs docker + nvidia-smi):

  python3 bench/ab_bench.py --label base-27b                      # 27B, defaults
  python3 bench/ab_bench.py --label base-moe --model qwen3.6-35b-a3b --streams 4,8,16
  python3 bench/ab_bench.py --label x --save-reference            # store the correctness reference

Talks to the SGLang server started by scripts/launch.sh (default http://127.0.0.1:30000; SGL_BASE or --base to
change) and reads that container's log for server-side throughput. Keys are read from $SGL_ROOT (default
/opt/sglang). Measures, in this order, after waiting for 10 s of no running/queued requests:
  1. decode, 1 stream: three 600-token greedy replies; the number reported is the MEDIAN of the scheduler's own
     'gen throughput' for batches with one running request (client numbers read 10-15% low: sleep-on-idle ramp).
  2. decode, N streams: N parallel 400-token replies per level; server-side aggregate median for that batch size.
  3. prefill: uncached prompts (labels 8000 and 32000 = about 6K and 24K tokens; a unique first line defeats the
     prefix cache), max_tokens 1; tokens / seconds-to-completion, median of 2.
  4. TTFT at the highest stream count (client side, includes queueing).
  5. energy: GPU board power sampled every 0.5 s during steps 1-2 -> joules per output token.
  6. correctness vs the stored reference for this model: prompt log-prob fingerprint of a fixed ~1,500-token
     passage (mean and max absolute difference) and the greedy 160-token text (common-prefix length).
     Bit-identical is expected for launcher-only changes; kernel/backend changes may move logprobs slightly
     (mean |d| well under 0.02 and a long common prefix is normal; a large mean or an early divergence is not).
Results append one JSON line to $SGL_ROOT/ab/results.jsonl and print a short table.
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time, urllib.request

ROOT = os.environ.get("SGL_ROOT", "/opt/sglang")
P = argparse.ArgumentParser()
P.add_argument("--label", required=True)
P.add_argument("--base", default=os.environ.get("SGL_BASE", "http://127.0.0.1:30000"))
P.add_argument("--model", default="qwen3.8-27b")
P.add_argument("--streams", default="2,5")
P.add_argument("--prefill", default="8000,32000")
P.add_argument("--container", default=os.environ.get("CONTAINER_NAME", "sgl-5090"))
P.add_argument("--out", default=f"{ROOT}/ab/results.jsonl")
P.add_argument("--save-reference", action="store_true")
P.add_argument("--skip-correctness", action="store_true")
A = P.parse_args()
KEY = open(f"{ROOT}/api_key").read().strip()
REF = f"{ROOT}/ab/reference_{A.model}.json"
TOPICS = ["how DDR5 memory training works at boot", "why suspension bridges oscillate in wind",
          "how a heat pump moves heat uphill", "the history of the printing press in Asia",
          "how ocean currents redistribute heat", "how compilers allocate registers",
          "why concrete cracks as it cures", "how noise-cancelling headphones work",
          "how a jet engine starts", "the chemistry of sourdough bread",
          "how satellites keep their orbits", "why lithium cells age",
          "how river deltas form", "how elevators stay safe", "how vaccines train immunity",
          "how chess engines search"]


def req(path, body=None, timeout=900):
    r = urllib.request.Request(A.base + path, data=None if body is None else json.dumps(body).encode())
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + KEY)
    return urllib.request.urlopen(r, timeout=timeout)


def metric(name):
    try:
        with req("/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
        m = re.search(r"^" + re.escape(name) + r"(?:\{[^}]*\})? (\S+)", txt, re.M)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def wait_idle(limit=300):
    quiet, t0 = 0, time.time()
    while time.time() - t0 < limit:
        busy = (metric("sglang:num_running_reqs") or 0) + (metric("sglang:num_queue_reqs") or 0)
        quiet = quiet + 1 if busy == 0 else 0
        if quiet >= 5:
            return True
        time.sleep(2)
    return False


def chat(prompt, max_tokens, temperature=0.0):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    first, usage = None, None
    with req("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                if first is None and (c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")):
                    first = time.perf_counter()
    t1 = time.perf_counter()
    return {"ttft": (first or t1) - t0, "total": t1 - t0,
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0), "completion_tokens": (usage or {}).get("completion_tokens", 0)}


class Power(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def gen_tp_since(epoch, running):
    """Median server-side 'gen throughput' for decode batches with exactly `running` requests since epoch."""
    out = subprocess.run(["docker", "logs", "--since", str(int(epoch)), A.container], capture_output=True, text=True).stdout
    out += subprocess.run(["docker", "logs", "--since", str(int(epoch)), A.container], capture_output=True, text=True).stderr
    vals = []
    for line in out.splitlines():
        m = re.search(r"Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([0-9.]+)", line)
        if m and int(m.group(1)) == running:
            vals.append(float(m.group(2)))
    vals = vals[1:] if len(vals) > 3 else vals  # first sample includes the clock ramp
    return (round(statistics.median(vals), 1), len(vals)) if vals else (None, 0)


def words(n, salt):
    base = ("bandwidth weights token decode prefill cache memory lane clock voltage curve stable error pattern "
            "kernel batch latency throughput window context slot state radix prefix river bridge engine").split()
    x, out = salt * 7919 + 13, []
    for i in range(n):
        x = (x * 1103515245 + 12345) % (2 ** 31)
        out.append(base[x % len(base)])
    return " ".join(out)


def passage():
    out, x = [], 12345
    ws = ("bandwidth weights token decode prefill cache memory lane clock voltage curve stable error "
          "pattern kernel batch latency throughput window context slot state radix prefix").split()
    for i in range(1100):
        x = (x * 1103515245 + 12345) % (2 ** 31)
        out.append(ws[x % len(ws)])
        if i % 13 == 12:
            out[-1] += "."
    return "Section one. " + " ".join(out)


def flush_cache():
    admin = open(f"{ROOT}/admin_api_key").read().strip()
    r = urllib.request.Request(A.base + "/flush_cache", data=b"{}", method="POST")
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + admin)
    with urllib.request.urlopen(r, timeout=60) as f:
        return f.status == 200


def correctness():
    # Flush first: whether the passage's prefix is served from the radix cache (bf16 state checkpoints) shifts the
    # logprobs by ~0.16 nats on average (measured 2026-09-23). After a flush the probe is bit-identical run to run.
    flush_cache()
    # Only the LAST 128 prompt positions get logprobs. logprob_start_len 0 on a ~1,600-token passage makes the
    # server materialise ~1,600 x 248,320 fp32 logits (~1.6 GB, twice with log_softmax); PyTorch's caching
    # allocator keeps that memory, which pushed an RTX 5090 under WSL2 into WDDM paging (decode 86 -> 36 tok/s).
    with req("/generate", {"text": passage(), "sampling_params": {"max_new_tokens": 1, "temperature": 0}}) as r:
        n = json.load(r)["meta_info"]["prompt_tokens"]
    with req("/generate", {"text": passage(), "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": max(0, n - 128)}) as r:
        g = json.load(r)
    lp = [x[0] for x in g["meta_info"]["input_token_logprobs"] if x[0] is not None]
    with req("/generate", {"text": "Explain step by step why the sky is blue, then list three related phenomena.",
                           "sampling_params": {"max_new_tokens": 160, "temperature": 0}}) as r:
        text = json.load(r)["text"]
    return lp, text


def main():
    res = {"label": A.label, "model": A.model, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["pool"] = metric("sglang:max_total_num_tokens")
    try:
        args = subprocess.run(["docker", "inspect", A.container, "--format", "{{join .Args \" \"}}"],
                              capture_output=True, text=True).stdout
        res["args"] = re.sub(r"[0-9a-f]{48}", "<KEY>", args.strip())
    except Exception:
        pass
    if not wait_idle():
        print("WARNING: server never idle for 10 s; numbers may include other traffic")
        res["not_idle"] = True
    pw = Power(); pw.start()
    t_dec = time.time()
    single = [chat(f"Write a long, detailed technical essay about {TOPICS[i]}.", 600) for i in range(3)]
    res["decode1_server"], res["decode1_samples"] = gen_tp_since(t_dec, 1)
    res["decode1_client"] = round(statistics.median(
        (s["completion_tokens"] - 1) / max(1e-6, s["total"] - s["ttft"]) for s in single), 1)
    out_tokens = sum(s["completion_tokens"] for s in single)
    t_conc_all, ttft_last = time.time(), None
    for n in [int(x) for x in A.streams.split(",") if x]:
        wait_idle(60)
        t_n, results, th = time.time(), [None] * n, []
        def one(i):
            results[i] = chat(f"Write three detailed paragraphs about {TOPICS[i % len(TOPICS)]} (variant {i}).", 400)
        for i in range(n):
            th.append(threading.Thread(target=one, args=(i,))); th[-1].start()
        for t in th:
            t.join()
        res[f"decode{n}_server"], res[f"decode{n}_samples"] = gen_tp_since(t_n, n)
        out_tokens += sum(r["completion_tokens"] for r in results if r)
        ttft_last = max(r["ttft"] for r in results if r)
        res[f"ttft_max_at_{n}"] = round(ttft_last, 3)
    pw.stop.set(); pw.join(timeout=3)
    dec_seconds = time.time() - t_dec
    if pw.samples:
        watts = statistics.mean(pw.samples)
        res["power_w_mean"] = round(watts, 1)
        res["joules_per_out_token"] = round(watts * dec_seconds / max(1, out_tokens), 3)
    for L in [int(x) for x in A.prefill.split(",") if x]:
        vals = []
        for rep in range(2):
            wait_idle(60)
            salt = int(time.time() * 1000) % 1000003
            prompt = f"Unique run id {salt}-{rep}. Summarise the following text in one sentence.\n" + words(int(L * 0.72), salt)
            r = chat(prompt, 1)
            vals.append((r["prompt_tokens"], r["total"]))
        pt = statistics.median(v[0] for v in vals)
        res[f"prefill_{L}_tokens"] = int(pt)
        res[f"prefill_{L}_tok_s"] = round(statistics.median(v[0] / v[1] for v in vals), 0)
    if not A.skip_correctness:
        wait_idle(60)
        lp, text = correctness()
        if A.save_reference:
            json.dump({"logprobs": lp, "text": text, "label": A.label}, open(REF, "w"))
            res["reference_saved"] = REF
        elif os.path.exists(REF):
            ref = json.load(open(REF))
            n = min(len(lp), len(ref["logprobs"]))
            d = [abs(lp[i] - ref["logprobs"][i]) for i in range(n)]
            res["fp_mean_abs_diff"] = round(statistics.mean(d), 5)
            res["fp_max_abs_diff"] = round(max(d), 4)
            res["fp_identical"] = lp == ref["logprobs"]
            k = 0
            while k < min(len(text), len(ref["text"])) and text[k] == ref["text"][k]:
                k += 1
            res["greedy_common_prefix_chars"] = k
            res["greedy_ref_chars"] = len(ref["text"])
        else:
            res["correctness"] = "no reference yet (run with --save-reference)"
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    keys = [k for k in res if k not in ("args",)]
    w = max(len(k) for k in keys)
    for k in keys:
        print(f"{k:<{w}}  {res[k]}")


if __name__ == "__main__":
    main()
