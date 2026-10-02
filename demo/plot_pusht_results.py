"""Generate a presentation-quality PNG of the PushT budget sweep results."""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LOG = Path(__file__).parent.parent / "logs" / "pusht-sweep-22990492.out"
OUT = Path(__file__).parent / "pusht_sweep_results.png"

# ── Parse log ─────────────────────────────────────────────────────────────────

results: dict[int, dict[str, list[float]]] = {}
current_budget = None

with open(LOG) as f:
    for line in f:
        m = re.search(r"Budget = (\d+)", line)
        if m:
            current_budget = int(m.group(1))
            results.setdefault(current_budget, {"puct": [], "gumbel": []})
            continue
        m = re.search(r"puct=([\d.]+)\s+gumbel=([\d.]+)", line)
        if m and current_budget:
            results[current_budget]["puct"].append(float(m.group(1)))
            results[current_budget]["gumbel"].append(float(m.group(2)))

budgets = sorted(results)
n_seeds = len(results[budgets[0]]["puct"])

# ── Compute stats ─────────────────────────────────────────────────────────────

def stats(vals):
    a = np.array(vals)
    mean = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    return mean, se

puct_means, puct_ses = zip(*[stats(results[b]["puct"]) for b in budgets])
gumbel_means, gumbel_ses = zip(*[stats(results[b]["gumbel"]) for b in budgets])

puct_means = np.array(puct_means)
puct_ses = np.array(puct_ses)
gumbel_means = np.array(gumbel_means)
gumbel_ses = np.array(gumbel_ses)

# ── Plot ──────────────────────────────────────────────────────────────────────

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(budgets))
w = 0.32

bars_puct = ax.bar(x - w / 2, puct_means, w, yerr=puct_ses,
                   color=ORANGE, capsize=3, error_kw=dict(lw=1.2, color=TEXT_SECONDARY),
                   edgecolor=SURFACE, linewidth=0.5, label="PUCT (Q = 0)", zorder=3)
bars_gumbel = ax.bar(x + w / 2, gumbel_means, w, yerr=gumbel_ses,
                     color=BLUE, capsize=3, error_kw=dict(lw=1.2, color=TEXT_SECONDARY),
                     edgecolor=SURFACE, linewidth=0.5, label="Gumbel", zorder=3)

# Direct labels on Gumbel bars
for i, (mean, se) in enumerate(zip(gumbel_means, gumbel_ses)):
    ax.text(x[i] + w / 2, mean + se + 0.008, f"{mean:.3f}",
            ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY,
            fontfamily="sans-serif")

ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in budgets], fontsize=10, color=TEXT_PRIMARY)
ax.set_xlabel("Simulation budget", fontsize=11, color=TEXT_PRIMARY, labelpad=8)
ax.set_ylabel("Final IoU", fontsize=11, color=TEXT_PRIMARY, labelpad=8)

ax.set_ylim(0, max(gumbel_means + gumbel_ses) * 1.35)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_PRIMARY, labelsize=9)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

ax.set_title("PushT — PUCT vs Gumbel MCTS\n",
             fontsize=13, fontweight="bold", color=TEXT_PRIMARY, pad=4,
             fontfamily="sans-serif")
ax.text(0.5, 1.02,
        f"Rollout depth 3  ·  Diffusion prior  ·  {n_seeds} seeds  ·  mean ± SE",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.5, color=TEXT_SECONDARY, fontfamily="sans-serif")

legend = ax.legend(loc="upper left", frameon=True, fontsize=9,
                   edgecolor=GRIDLINE, facecolor=SURFACE, framealpha=1)
for text in legend.get_texts():
    text.set_color(TEXT_PRIMARY)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print(f"Saved {OUT}")
