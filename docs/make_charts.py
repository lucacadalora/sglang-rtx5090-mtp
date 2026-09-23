#!/usr/bin/env python3
"""Regenerate the charts in docs/img from the measured points (bench/results) and the roofline constants.

  python docs/make_charts.py        (needs matplotlib)

Model constants (see docs/REPORT.md, section 2): W = 14.5 GB of weights read per decode step, BW = 1.40 TB/s effective
(calibrated to the measured 96 tok/s; rated 1,792 GB/s), S = 74.8 MiB of Gated DeltaNet state per running request,
k = 32 KiB of FP8 KV per context token, 2N = 53 GFLOP per token, F = 40% of 1,676 dense FP4 TFLOPS.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#888780"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})

W, BW, S, K = 14.5e9, 1.40e12, 74.8 * 2**20, 32 * 1024
T0 = W / BW * 1e3                      # ms
COMPUTE = 53e9 / (0.4 * 1.676e15) * 1e3  # ms per token


def m(ctx):
    """Memory time per extra running request per step, ms."""
    return (S + ctx * K) / BW * 1e3


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. roofline: step time and GPU time per token vs requests decoding together
B = [2**i for i in range(11)]
meas_short = [(1, 10.70, 10.42, 11.15), (2, 11.37, 11.06, 11.70), (5, 10.96, 10.52, 11.61)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, per_token in ((a1, False), (a2, True)):
    div = (lambda b: b) if per_token else (lambda b: 1)
    ax.plot(B, [(T0 + b * m(1024)) / div(b) for b in B], color=BLUE, lw=2, label="Memory, 1K context")
    b60 = [b for b in B if b <= 64]
    ax.plot(b60, [(T0 + b * m(61440)) / div(b) for b in b60], color=ORANGE, lw=2, label="Memory, 60K context")
    ax.plot(B, [(b * COMPUTE) / div(b) for b in B], color=GRAY, lw=2, ls="--", label="Compute, 40% of FP4 peak")
    for b, ms, lo, hi in meas_short:
        v, l, h = (ms / b, lo / b, hi / b) if per_token else (ms, lo, hi)
        ax.errorbar([b], [v], yerr=[[v - l], [h - v]], fmt="o", color=INK, ms=5, capsize=3,
                    label="Measured, short context" if b == 1 else None)
    ax.plot([1], [1000 / 85], "D", color=ORANGE, mec=INK, ms=6, label="Measured, 61K context")
    ax.axvline(5, color=MUTED, ls=":", lw=1)
    ax.text(5.4, 0.93, "our cap: 5 running", color=MUTED, fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_xlabel("Requests decoding together (B)")
    ax.set_ylabel("GPU time per output token (ms)" if per_token else "Time per decode step (ms)")
    ax.set_title("Cost per token (step time / B)" if per_token else "Time per decode step")
a1.legend(loc="lower right", fontsize=9)
save(fig, "roofline.svg")

# 2. VRAM budget
parts = ["Weights", "KV pool", "GDN state slots", "Activations and CUDA graphs", "Free headroom", "Desktop apps",
         "Driver reserve and misc"]
colors = [GRAY, BLUE, VIOLET, YELLOW, AQUA, ORANGE, "#c3c2b7"]
configs = {
    "Plain 27B": [17.10, 4.73, 1.68, 2.54, 1.75, 3.50, 0.54],
    "MTP, embedding on GPU\n(window 101,376)": [17.82, 3.63, 1.68, 2.93, 1.75, 3.50, 0.54],
    "MTP + embedding in RAM\n(now, window 147,456)": [15.31, 6.14, 1.68, 2.93, 1.75, 3.50, 0.54],
}
fig, ax = plt.subplots(figsize=(10, 3.4))
names = list(configs)[::-1]
for yi, name in enumerate(names):
    left = 0
    for pi, val in enumerate(configs[name]):
        ax.barh(yi, val, left=left, color=colors[pi], edgecolor="white", lw=1.5, height=0.6,
                label=parts[pi] if yi == 0 else None)
        if val >= 1.5:
            ax.text(left + val / 2, yi, f"{val:.1f}", ha="center", va="center", color="white" if pi in (0, 1, 3) else INK,
                    fontsize=9)
        left += val
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("VRAM (GiB of the 31.8 GiB card)")
ax.set_title("Where the 32 GB goes (KV pool 155,064 / 111,953 / 189,329 tokens, top to bottom)")
ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_xlim(0, 32)
save(fig, "vram_budget.svg")

# 3. paging incident
phases = ["Healthy", "Paging\n(0.5 GB free)", "After\n/flush_cache", "After\nrestart"]
tps = [(96, 92, 99), (15, 15, 15), (75, 73, 76), (96, 92, 99)]
pcie = [(0.05, 0.015, 0.08), (10.3, 5.2, 16.7), (2.5, 1.7, 2.7), (0.05, 0.015, 0.08)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, data, color, title, unit in ((a1, tps, BLUE, "Single-stream decode", "tok/s"),
                                     (a2, pcie, ORANGE, "PCIe read during decode", "GB/s")):
    x = range(len(phases))
    vals = [d[0] for d in data]
    ax.bar(x, vals, color=color, width=0.55)
    ax.errorbar(x, vals, yerr=[[d[0] - d[1] for d in data], [d[2] - d[0] for d in data]], fmt="none", ecolor=INK,
                capsize=3, lw=1)
    for xi, d in zip(x, data):
        ax.text(xi + 0.3, d[0], f"{d[0]:g}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), phases)
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
fig.suptitle("WDDM paging incident, reproduced on purpose (whiskers: range of 1 s samples)", color=MUTED, fontsize=10)
save(fig, "paging.svg")

# 4. decode vs context length
ctx = [i * 16384 for i in range(10)]
plain = [1000 / (T0 + m(c)) for c in ctx]
dense = [1000 / (T0 + c * 4 * K / BW * 1e3) for c in ctx]
cyc0 = 2.57 / 152.25 * 1000 - 1024 * 38 * 1024 / BW * 1e3
mtp = [2.57 / (cyc0 + c * 38 * 1024 / BW * 1e3) * 1000 for c in ctx]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(ctx, mtp, color=BLUE, lw=2, label="Model, MTP at 2.57 tokens per cycle")
ax.plot(ctx, plain, color=ORANGE, lw=2, label="Model, plain 27B (32 KiB per token)")
ax.plot(ctx, dense, color=GRAY, lw=2, ls="--", label="Model, if all 64 layers were attention")
for (c, v, lo, hi), col, mk, lab in [((1000, 93.5, 89.7, 96.0), ORANGE, "o", "Plain, measured"),
                                     ((61000, 85, 85, 85), ORANGE, "o", None),
                                     ((1000, 152.3, 149.3, 164.3), BLUE, "s", "MTP, measured"),
                                     ((61000, 119, 117, 121), BLUE, "s", None),
                                     ((129000, 92.5, 92.5, 92.5), BLUE, "s", None)]:
    ax.errorbar([c], [v], yerr=[[v - lo], [hi - v]], fmt=mk, color=col, mec=INK, ms=6, capsize=3, label=lab)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Decode, one stream (tok/s)")
ax.set_ylim(0, 175)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
ax.grid(True, color=GRID, lw=0.8)
ax.set_title("Single-stream decode vs context (short points greedy; 61K and 129K sampled at temperature 0.6)")
ax.legend(fontsize=9, loc="lower left")
save(fig, "decode_vs_context.svg")

# 5. throughput vs concurrent streams
series = {
    "27B + MTP (now)": (BLUE, [(1, 152.3, 149.3, 164.3), (2, 290.9, 278.1, 303.7), (5, 752.6, 692.1, 767.9)]),
    "27B plain": (ORANGE, [(1, 93.5, 89.7, 96.0), (2, 175.9, 170.9, 180.9), (5, 456.3, 430.6, 475.2)]),
    "Qwen3.6-35B-A3B MoE (1 run)": (AQUA, [(1, 303.1, 303.1, 303.1), (4, 807.9, 807.9, 807.9), (8, 1320.7, 1320.7, 1320.7)]),
}
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, (col, pts) in series.items():
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=col, lw=2)
    ax.errorbar(xs, ys, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], fmt="o", color=col, mec=INK,
                ms=5, capsize=3, label=name)
ax.set_xlabel("Concurrent streams")
ax.set_ylabel("Total decode (tok/s)")
ax.set_title("Total decode throughput (medians of 2-4 runs; whiskers = min to max)")
ax.grid(True, color=GRID, lw=0.8)
ax.set_ylim(0, 1400)
ax.legend(fontsize=9)
save(fig, "throughput.svg")

# 6. where the single-stream gain came from
labels = ["Before\n(paging-degraded)", "Paging fix", "MTP + embedding\nin RAM", "Kernel + runtime\nwork (estimate)",
          "Rated-bandwidth\nceiling"]
spans = [(0, 87), (87, 93), (93, 152), (152, 200), (0, 240)]
tags = ["87", "+6", "+59 = 152", "est. 185-215", "240"]
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, ((lo, hi), tag) in enumerate(zip(spans, tags)):
    if i == 3:
        ax.bar(i, hi - lo, bottom=lo, color="#b5d4f4", edgecolor=BLUE, ls="--", width=0.55)
    elif i == 4:
        ax.bar(i, hi - lo, bottom=lo, color="white", edgecolor=GRAY, width=0.55)
    else:
        ax.bar(i, hi - lo, bottom=lo, color=GRAY if i == 0 else BLUE, width=0.55)
    ax.text(i, hi + 4, tag, ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(range(len(labels)), labels)
ax.set_ylabel("Single-stream decode (tok/s)")
ax.set_ylim(0, 265)
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Where the single-stream gain came from, and what is left")
save(fig, "bridge.svg")
