"""Turn llama-bench JSON into a roofline table: decode tok/s vs bandwidth ceiling."""
import json, glob, sys, os
BW = float(sys.argv[1]) if len(sys.argv) > 1 else 36.5e9   # measured read bandwidth, kernels/bw.cu
BPT = json.load(open("results/bytes_per_token.json"))   # from scripts/bytes_per_token.py
rows = []
for f in sorted(glob.glob("results/raw/baseline-*.json")):
    try: data = json.load(open(f))
    except Exception: continue
    name = os.path.basename(f)[9:-5].replace("qwen2.5-", "").replace("-instruct", "")
    pp = next((r for r in data if r["n_prompt"] > 0 and r["n_gen"] == 0), None)
    tg = next((r for r in data if r["n_gen"] > 0 and r["n_prompt"] == 0), None)
    size = tg["model_size"]
    key = os.path.basename(f)[9:-5]
    ceil = BW / BPT[key]["bytes_per_token"]   # all weights except token_embd (one row per token)
    rows.append((name, size / 2**20, pp["avg_ts"], tg["avg_ts"], tg["stddev_ts"], ceil, 100 * tg["avg_ts"] / ceil))
out = ["| model | file MiB | prefill tok/s (pp512) | decode tok/s (tg128) | ceiling tok/s (36.5 GB/s ÷ bytes/token) | % of ceiling |",
       "|---|---|---|---|---|---|"]
for n, s, pp, tg, sd, c, pct in rows:
    out.append(f"| {n} | {s:.0f} | {pp:.0f} | {tg:.1f} ± {sd:.1f} | {c:.1f} | {pct:.0f}% |")
print("\n".join(out))
