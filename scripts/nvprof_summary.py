"""Summarise an nvprof --print-gpu-summary CSV: time per decoded token by kernel family."""
import csv, sys, re
path, ntok = sys.argv[1], int(sys.argv[2])
rows = [r for r in csv.reader(open(path)) if len(r) > 7 and r[0] == "GPU activities"]
fam = {}
for r in rows:
    t = float(r[2]); name = r[7]
    m = re.match(r"(?:void )?([A-Za-z_0-9]+)(<[^>]*>)?", name)
    short = m.group(1) + (m.group(2) or "") if m else name[:60]
    short = short.replace("ggml_type=2", "Q4_0").replace("ggml_type=8", "Q8_0").replace("ggml_type=14", "Q6_K")
    fam[short] = fam.get(short, [0.0, 0]); fam[short][0] += t; fam[short][1] += int(r[3])
total = sum(v[0] for v in fam.values())
print(f"total GPU kernel time {total*1e3:.0f} ms over the whole run ({ntok} tokens + prompt)")
print("| kernel | ms per token | % of GPU time | calls per token |\n|---|---|---|---|")
for k, (t, c) in sorted(fam.items(), key=lambda x: -x[1][0])[:12]:
    print(f"| `{k[:70]}` | {t/ntok*1e3:.2f} | {100*t/total:.1f}% | {c/ntok:.0f} |")
