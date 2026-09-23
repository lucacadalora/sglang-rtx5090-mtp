#!/usr/bin/env bash
# Mean MTP accept length and per-window numbers from the server's decode log (read-only: docker logs).
#
# Usage (WSL, as root):
#   bash accept_len.sh SINCE [UNTIL] [--windows] [--csv FILE]
#     SINCE / UNTIL  anything `docker logs --since/--until` takes: 15m, 2h, 2026-09-24T10:00:00 (UTC),
#                    2026-09-24T17:00:00+07:00, or a unix timestamp.
#     --windows      also print every window (one "Decode batch" line = decode_log_interval (40) iterations).
#     --csv FILE     write the parsed windows as CSV.
#   CONTAINER=name overrides the container (default sgl-5090, the name scripts/launch.sh uses).
#
# Per window: running requests, accept len, accept rate, gen throughput and
#   cycle_ms = accept_len * running / gen_throughput * 1000   (the plan's bs=1 cycle metric).
# The first window of a burst (more than 3 s after the previous window) includes idle time in its throughput,
# so it is excluded from the cycle statistics (it still counts for accept length).
# Only lines matching "Decode batch" with "accept len:" are read, and only parsed numbers are printed; never
# raw log lines (the server_args log line contains the API keys).
set -euo pipefail
CONTAINER="${CONTAINER:-sgl-5090}"
if [ $# -lt 1 ]; then sed -n '2,17p' "$0"; exit 2; fi
SINCE="$1"; shift
UNTIL=""
WINDOWS=0
CSV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --windows) WINDOWS=1 ;;
    --csv) CSV="$2"; shift ;;
    *) UNTIL="$1" ;;
  esac
  shift
done
ARGS=(--since "$SINCE")
[ -n "$UNTIL" ] && ARGS+=(--until "$UNTIL")
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
docker logs "${ARGS[@]}" "$CONTAINER" 2>&1 | grep -E 'Decode batch.*accept len: ' > "$TMP" || true
WINDOWS="$WINDOWS" CSV="$CSV" python3 - "$TMP" <<'EOF'
import csv, datetime, os, re, statistics, sys
pat = re.compile(
    r"^\[(?P<ts>[0-9-]+ [0-9:]+)\] Decode batch, #running-req: (?P<run>\d+).*?"
    r"accept len: (?P<al>[0-9.]+), accept rate: (?P<ar>[0-9.]+), cuda graph: (?P<cg>\w+), "
    r"gen throughput \(token/s\): (?P<tp>[0-9.]+)")
rows = []
prev = None
for line in open(sys.argv[1], errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    ts = datetime.datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")
    run, al, ar, tp = int(m["run"]), float(m["al"]), float(m["ar"]), float(m["tp"])
    burst_start = prev is None or (ts - prev).total_seconds() > 3
    prev = ts
    cyc = al * run / tp * 1000 if tp > 0 and run > 0 else float("nan")
    rows.append(dict(ts=m["ts"], running=run, accept_len=al, accept_rate=ar, cuda_graph=m["cg"],
                     gen_tok_s=tp, cycle_ms=round(cyc, 3), burst_start=burst_start))
if not rows:
    print("no 'Decode batch ... accept len' lines in that interval")
    sys.exit(1)
if os.environ.get("CSV"):
    with open(os.environ["CSV"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} windows to {os.environ['CSV']}")
if os.environ.get("WINDOWS") == "1":
    print(f"{'time':19s} {'run':>3s} {'acc_len':>7s} {'acc_rate':>8s} {'tok/s':>8s} {'cycle_ms':>8s} graph")
    for r in rows:
        mark = "  (burst start, excluded from cycle stats)" if r["burst_start"] else ""
        print(f"{r['ts']:19s} {r['running']:3d} {r['accept_len']:7.2f} {r['accept_rate']:8.2f} "
              f"{r['gen_tok_s']:8.1f} {r['cycle_ms']:8.2f} {r['cuda_graph']}{mark}")

def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]

print(f"windows: {len(rows)}  ({rows[0]['ts']} .. {rows[-1]['ts']}, container-local time)")
w_all = sum(r["running"] for r in rows)
print(f"accept len: mean {statistics.mean(r['accept_len'] for r in rows):.3f} (per window), "
      f"{sum(r['accept_len'] * r['running'] for r in rows) / w_all:.3f} (weighted by running requests), "
      f"median {statistics.median(r['accept_len'] for r in rows):.2f}")
print(f"{'running':>7s} {'windows':>7s} {'acc_len':>7s} {'acc_rate':>8s} {'tok/s':>8s} "
      f"{'cycle_ms p10/med/p90 (steady windows)':>40s}")
for run in sorted({r["running"] for r in rows}):
    g = [r for r in rows if r["running"] == run]
    s = [r["cycle_ms"] for r in g if not r["burst_start"]]
    cyc = f"{pct(s, 0.1):.2f} / {statistics.median(s):.2f} / {pct(s, 0.9):.2f}  (n={len(s)})" if s else "n/a"
    print(f"{run:7d} {len(g):7d} {statistics.mean(r['accept_len'] for r in g):7.3f} "
          f"{statistics.mean(r['accept_rate'] for r in g):8.3f} {statistics.median(r['gen_tok_s'] for r in g):8.1f} "
          f"{cyc:>40s}")
if any(r["cuda_graph"] != "True" for r in rows):
    print(f"note: {sum(r['cuda_graph'] != 'True' for r in rows)} windows ran without CUDA graphs")
EOF
