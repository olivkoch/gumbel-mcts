"""Generate a presentation-quality table PNG of the PushT sweep results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "demo/png/pusht_results_table.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_LIGHT = "#cde2fb"
ORANGE_LIGHT = "#fde0d0"
HIGHLIGHT = "#d6eaff"

budgets = [16, 32, 64, 128, 256]
puct_success = ["0%", "0%", "0%", "0%", "0%"]
puct_mean = ["0.020", "0.011", "0.023", "0.006", "0.004"]
gumbel_success = ["4%", "4%", "10%", "12%", "4%"]
gumbel_mean = ["0.083", "0.082", "0.109", "0.151", "0.113"]

fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.axis("off")

col_labels = ["Budget", "Success", "Mean IoU", "Success", "Mean IoU"]
cell_text = []
for i in range(len(budgets)):
    cell_text.append([
        str(budgets[i]),
        puct_success[i],
        puct_mean[i],
        gumbel_success[i],
        gumbel_mean[i],
    ])

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.0, 1.0, 0.82],
)

table.auto_set_font_size(False)
table.set_fontsize(11)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(GRIDLINE)
    cell.set_linewidth(0.8)
    cell.set_text_props(color=TEXT_PRIMARY, fontfamily="sans-serif")

    if row == 0:
        if col <= 2:
            cell.set_facecolor(ORANGE_LIGHT)
        else:
            cell.set_facecolor(BLUE_LIGHT)
        cell.set_text_props(fontweight="bold", fontsize=10, color=TEXT_PRIMARY)
        cell.set_height(0.18)
    else:
        cell.set_facecolor(SURFACE)
        cell.set_height(0.16)
        if col == 0:
            cell.set_text_props(fontweight="bold")
        # Highlight the best row
        if budgets[row - 1] == 128 and col >= 3:
            cell.set_text_props(fontweight="bold", color=BLUE)

# Column group headers
ax.text(0.20, 0.88, "PUCT (Q = 0)", ha="center", va="center",
        fontsize=12, fontweight="bold", color=ORANGE,
        transform=ax.transAxes, fontfamily="sans-serif")
ax.text(0.70, 0.88, "Gumbel MCTS", ha="center", va="center",
        fontsize=12, fontweight="bold", color=BLUE,
        transform=ax.transAxes, fontfamily="sans-serif")

ax.text(0.5, 0.99, "PushT — Final IoU  (50 seeds, rollout depth 3, diffusion prior)",
        ha="center", va="top", fontsize=11, color=TEXT_SECONDARY,
        transform=ax.transAxes, fontfamily="sans-serif")

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print(f"Saved {OUT}")
