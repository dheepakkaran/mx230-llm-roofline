"""Kernel table + projected 0.5B decode from results/raw/q4mv-v2.csv (kernels/q4mv.cu output)."""
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1] if len(sys.argv) > 1 else "results/raw/q4mv-v2.csv")) if r and not r[0].startswith("#") and r[0] != "shape"]
K = ["v1_naive", "v2_warp", "v3_dp4a", "v4_repack", "v5_repack2", "v6_smem_gguf", "v7_smem_gguf_2row"]
by = {}
for s, k, us, qx, gb, pct, err, llama in rows:
    by.setdefault(s, {})[k] = (float(us) + float(qx), float(pct), float(err), float(llama))
print("| shape | " + " | ".join(K) + " | llama.cpp (in-situ, unfused) |")
print("|---|" + "---|" * (len(K) + 1))
for s, d in by.items():
    cells = [f"{d[k][0]:.1f} µs ({d[k][1]:.0f}%)" for k in K]
    ll = next(iter(d.values()))[3]
    print(f"| {s.strip()} | " + " | ".join(cells) + f" | {ll:.1f} µs |" if ll else f"| {s.strip()} | " + " | ".join(cells) + " | – |")
# projection for one 0.5B token: per layer q,k,v,o,gate,up,down; x quantized 4x/layer (counted once per input)
names = {"qo": "0.5B attn q/o   896x896", "kv": "0.5B attn k/v   128x896", "up": "0.5B ffn up    4864x896", "down": "0.5B ffn down  896x4864"}
def per_layer(k):
    t = lambda n: float(next(r[2] for r in rows if r[0] == names[n] and r[1] == k))
    q = float(next(r[3] for r in rows if r[0] == names["qo"] and r[1] == k))
    return 2 * t("qo") + 2 * t("kv") + 2 * t("up") + t("down") + 4 * q
llama_layer = 2 * 26.4 + 2 * 5.0 + 2 * 127.0 + 106.1
print(f"\nper-layer matvec time, 0.5B: llama.cpp unfused {llama_layer:.1f} µs (in-situ)")
for k in ("v5_repack2", "v7_smem_gguf_2row"):
    pl = per_layer(k); save_ms = 24 * (llama_layer - pl) / 1e3
    tok_ms = 1000 / 59.7 - save_ms
    print(f"  {k}: {pl:.1f} µs/layer → saves {save_ms:.2f} ms/token → projected {1000/tok_ms:.1f} tok/s (fusion-off measured 59.7)")
