"""Greyscale figure for the README/page: decode tok/s per step vs the bandwidth ceiling.
Numbers are the measured ones from results/RESULTS.md (Steps 2, 4c, 6)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = ["llama.cpp\ndefault", "fusion off", "+ own kernel\n(stock layout)", "+ own kernel\n(repacked)"]
data = {"Qwen2.5-0.5B Q4_0": ([44.0, 60.1, 65.1, 75.0], 105.4),
        "Qwen2.5-1.5B Q4_0": ([19.7, 25.9, 25.5, 31.1], 39.3)}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
shades = ["#bdbdbd", "#8c8c8c", "#5a5a5a", "#1a1a1a"]
for ax, (title, (vals, ceil)) in zip(axes, data.items()):
    bars = ax.bar(range(4), vals, color=shades, width=0.62)
    ax.axhline(ceil, color="#1a1a1a", ls="--", lw=1)
    ax.text(3.45, ceil, f"bandwidth ceiling {ceil:.0f}", ha="right", va="bottom", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ceil * 0.015, f"{v:.1f}\n{100*v/ceil:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(4), steps, fontsize=7.5)
    ax.set_ylim(0, ceil * 1.18)
    ax.set_ylabel("decode tokens/s", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("GeForce MX230 (2 GB, Pascal): decode speed at each step", fontsize=11)
fig.tight_layout()
fig.savefig("docs/fig_speed.png")
print("wrote docs/fig_speed.png")
