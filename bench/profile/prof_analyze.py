#!/usr/bin/env python3
"""Summarise a torch-profiler chrome trace of decode steps: GPU busy vs idle, time by kernel category, top kernels.
Usage: python3 prof_analyze.py <trace.json[.gz]> [num_steps] [--json out.json]"""
import collections, gzip, json, re, sys

path = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    tr = json.load(f)
ev = tr["traceEvents"] if isinstance(tr, dict) else tr

gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu.sort(key=lambda e: e["ts"])
if not gpu:
    sys.exit("no GPU events in trace (CUPTI kernel tracing unavailable?)")

CATS = [
    ("gemm_fp4", r"cutlass|gemm|Gemm|GEMM|fp4|Fp4|nvfp4|sm120|Sm120|cublas|Kernel2|sgemm|hgemm|matmul|mma"),
    ("attention", r"flashinfer|BatchDecode|BatchPrefill|attention|Attention|fmha|xqa|merge_state|plan"),
    ("gdn_linear_attn", r"recurrent|gated_delta|delta_rule|chunk_|causal_conv|conv1d|fla_|_fwd_kernel|mamba|ssm|replay|Replay|gdn|GDN|l2norm|fused_gdn"),
    ("norm", r"rms|RMS|norm|Norm"),
    ("activation", r"silu|act_and_mul|gelu|swiglu|sigmoid"),
    ("quant", r"quant|Quant|cvt|scale_|fp8"),
    ("rope", r"rope|rotary|Rotary|mrope"),
    ("sampling_logits", r"softmax|argmax|sampl|top_k|topk|TopK|log_softmax|verify|tree|accept|eagle|spec"),
    ("embedding", r"embedding|host_embedding|gather"),
    ("elementwise_copy", r"elementwise|copy|Copy|vectorized|unrolled|fill|cat_|index|reduce|Reduce|where|add_|mul_|clamp"),
]


def cat_of(e):
    if e.get("cat") != "kernel":
        return e["cat"]
    n = e.get("name", "")
    for c, rx in CATS:
        if re.search(rx, n):
            return c
    return "other"


# busy time = union of kernel intervals
busy, cur_s, cur_e = 0.0, None, None
for e in gpu:
    s, d = e["ts"], e.get("dur", 0)
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, s + d
    else:
        cur_e = max(cur_e, s + d)
busy += cur_e - cur_s
span = gpu[-1]["ts"] + gpu[-1].get("dur", 0) - gpu[0]["ts"]

by_cat, by_name = collections.Counter(), collections.defaultdict(lambda: [0, 0.0])
for e in gpu:
    c = cat_of(e)
    by_cat[c] += e.get("dur", 0)
    k = (c, e.get("name", "")[:110])
    by_name[k][0] += 1
    by_name[k][1] += e.get("dur", 0)

# gaps between consecutive kernels (GPU idle bubbles), histogram
gaps = []
for a, b in zip(gpu, gpu[1:]):
    g = b["ts"] - (a["ts"] + a.get("dur", 0))
    if g > 0:
        gaps.append(g)
gaps.sort(reverse=True)

n = steps or 1
print(f"trace: {path}")
print(f"GPU events {len(gpu)} | span {span/1000:.2f} ms | busy {busy/1000:.2f} ms ({100*busy/span:.1f}%) | idle {(span-busy)/1000:.2f} ms")
if steps:
    print(f"per step ({steps} steps): span {span/1000/n:.2f} ms, busy {busy/1000/n:.2f} ms, kernels {len(gpu)/n:.0f}")
print("\n== GPU time by category ==")
tot = sum(by_cat.values())
for c, d in by_cat.most_common():
    print(f"  {c:<18} {d/1000:9.2f} ms  {100*d/tot:5.1f}%" + (f"   {d/1000/n:7.3f} ms/step" if steps else ""))
print("\n== top 45 kernels by total time ==")
for (c, name), (cnt, d) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:45]:
    print(f"  {d/1000:8.2f} ms  n={cnt:<6} avg {d/max(cnt,1):8.1f} us  [{c}] {name}")
print("\n== largest idle gaps (us) ==", [round(g, 1) for g in gaps[:15]])
print("gaps > 50 us:", sum(1 for g in gaps if g > 50), "totalling", round(sum(g for g in gaps if g > 50) / 1000, 2), "ms")
if out_json:
    json.dump({"span_us": span, "busy_us": busy, "steps": steps, "by_cat": dict(by_cat),
               "top": [[c, nm, cnt, d] for (c, nm), (cnt, d) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:200]],
               "gaps_top": gaps[:200]}, open(out_json, "w"))
