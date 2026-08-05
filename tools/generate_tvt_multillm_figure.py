"""Multi-LLM figure sized for single-column TVT placement."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper2/generated_figures/fig_multi_llm_generalization.pdf")
models = ["Qwen3-8B", "Qwen2.5-7B", "Qwen3.5-4B", "Grok-4.5", "GPT-5.6"]
rgd = [0.333, 0.333, 0.250, 0.333, 0.333]
fast = [0.250, 0.250, 0.250, 0.250, 0.250]
slow = [0.030, 0.011, 0.024, 0.027, 0.030]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
})

fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.55), gridspec_kw={"hspace": 0.42})
ax = axes[0]
x = np.arange(len(models))
w = 0.36
ax.bar(x - w / 2, rgd, w, color="#0072B2", edgecolor="white", label="RGD", linewidth=0.45)
ax.bar(x + w / 2, fast, w, color="#4D4D4D", edgecolor="white", label="Fast-only", linewidth=0.45)
ax.set_xticks(x, models, rotation=20, ha="right")
ax.set_ylabel("Completion")
ax.set_ylim(0.0, 0.48)
ax.legend(frameon=False, loc="upper right", fontsize=6.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.4)
ax.text(-0.18, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8.5)

ax = axes[1]
ax.scatter(slow, rgd, s=52, c="#0072B2", marker="o", edgecolor="white", linewidth=0.55, zorder=3, label="RGD")
ax.axhline(0.250, color="#4D4D4D", linestyle="--", linewidth=0.85, label="Fast-only")
for i, m in enumerate(models):
    ax.annotate(m, (slow[i], rgd[i]), textcoords="offset points", xytext=(4, 4), fontsize=5.8, color="#0072B2")
ax.set_xlabel("Slow-call rate")
ax.set_ylabel("Completion")
ax.set_xlim(0.005, 0.040)
ax.set_ylim(0.22, 0.40)
ax.legend(frameon=False, fontsize=6.2, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(color="#D9D9D9", linewidth=0.4)
ax.text(-0.18, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=8.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=400)
plt.close(fig)
print(OUT)
