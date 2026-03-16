"""
demo_visual_gomoku.py — Visit-distribution heatmap comparing PUCT vs Gumbel MCTS.

Runs PUCT and Gumbel Dense on curated 9×9 Gomoku positions and renders
side-by-side heatmaps showing how each algorithm distributes its search
budget.  Gumbel concentrates visits on the best candidates via sequential
halving, while PUCT spreads visits broadly.

Usage:
    uv run python examples/demo_visual_gomoku.py                              # default (64 sims, random model)
    uv run python examples/demo_visual_gomoku.py --sims 8                     # fewer sims
    uv run python examples/demo_visual_gomoku.py --model heuristic --sims 32  # noisy heuristic model
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from numba import njit

from gumbel_mcts import PUCT, GumbelDense

# ── Styling ──────────────────────────────────────────────────────────────────

PLAYER_COLORS = {1: "#222222", 2: "#EEEEEE"}
PLAYER_EDGE   = {1: "#000000", 2: "#999999"}
ALGO_COLORS   = {"PUCT": "#5C6BC0", "Gumbel Dense": "#26A69A"}
BOARD_COLOR   = "#DCB35C"
GRID_COLOR    = "#5D4037"
HEATMAP_CMAP  = "YlOrRd"

# ── 9×9 Gomoku game logic (5-in-a-row) ──────────────────────────────────────

BS9 = 9
NA9 = BS9 * BS9  # 81


@njit(cache=False)
def _count_dir9(board, r, c, dr, dc, player):
    count = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < BS9 and 0 <= cc < BS9 and board[rr, cc] == player:
        count += 1
        rr += dr
        cc += dc
    return count


@njit(cache=False)
def _check_win9(board, r, c, player):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        total = 1 + _count_dir9(board, r, c, dr, dc, player) + \
                    _count_dir9(board, r, c, -dr, -dc, player)
        if total >= 5:
            return True
    return False


@njit(cache=False)
def _fast_step9(board, action, player):
    r = action // BS9
    c = action % BS9
    board[r, c] = player
    if _check_win9(board, r, c, player):
        return 1.0, player, True, board
    all_filled = True
    for i in range(BS9):
        for j in range(BS9):
            if board[i, j] == 0:
                all_filled = False
                break
        if not all_filled:
            break
    if all_filled:
        return 0.0, 0, True, board
    return 0.0, 0, False, board


@njit(cache=False)
def _get_valid_mask9(board, player):
    mask = np.zeros(NA9, dtype=np.float32)
    for r in range(BS9):
        for c in range(BS9):
            if board[r, c] == 0:
                mask[r * BS9 + c] = 1.0
    return mask


class Gomoku9Logic:
    """9×9 Gomoku with 5-in-a-row to win."""
    GAME_NAME = "gomoku_9x9"
    NUM_ACTIONS = NA9
    BOARD_SHAPE = (BS9, BS9)
    MAX_MOVES = NA9
    MAX_LEGAL_MOVES = NA9
    PLAYER_1 = 1
    PLAYER_2 = 2
    NODE_STORAGE_WIDTH = NA9
    NN_OBS_WIDTH = NA9
    USE_HISTORY = False
    HISTORY_STEPS = 0
    fast_step = staticmethod(_fast_step9)
    get_valid_mask = staticmethod(_get_valid_mask9)

    @staticmethod
    def get_initial_board():
        return np.zeros((BS9, BS9), dtype=np.int8)


class RandomModel9(nn.Module):
    """Near-uniform random policy for 9×9 gomoku."""
    def __init__(self):
        super().__init__()
        self.logic = Gomoku9Logic()
        self.net = nn.Sequential(
            nn.Linear(NA9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.policy_head = nn.Linear(64, NA9)
        self.value_head = nn.Linear(64, 1)

    def forward_for_mcts(self, batch):
        x = batch["boards"].float()
        h = self.net(x.view(x.size(0), -1))
        policy = torch.softmax(self.policy_head(h) / 50.0, dim=-1)
        value = torch.tanh(self.value_head(h) * 0.01)
        return {"policy": policy, "value": value}


@njit(cache=False)
def _eval_board9(board, player):
    """Sliding-window heuristic evaluation for 9×9 (5-in-a-row)."""
    opponent = 3 - player
    score = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(BS9):
            for c in range(BS9):
                er = r + 4 * dr
                ec = c + 4 * dc
                if er < 0 or er >= BS9 or ec < 0 or ec >= BS9:
                    continue
                pc = 0
                oc = 0
                for k in range(5):
                    cell = board[r + k * dr, c + k * dc]
                    if cell == player:
                        pc += 1
                    elif cell == opponent:
                        oc += 1
                if oc == 0 and pc > 0:
                    if pc >= 5:   score += 100000.0
                    elif pc == 4: score += 5000.0
                    elif pc == 3: score += 500.0
                    elif pc == 2: score += 50.0
                    else:         score += 5.0
                if pc == 0 and oc > 0:
                    if oc >= 5:   score -= 100000.0
                    elif oc == 4: score -= 5000.0
                    elif oc == 3: score -= 500.0
                    elif oc == 2: score -= 50.0
                    else:         score -= 5.0
    return score


class HeuristicModel9(nn.Module):
    """Noisy threat-aware heuristic for 9×9 gomoku.

    Simulates a pretrained model that has learned to block/extend threats.
    Uses 1-ply lookahead (static eval after each candidate move) with heavy
    Gaussian noise so the prior is informative but far from perfect.
    """
    def __init__(self, noise_scale=2.0):
        super().__init__()
        self.logic = Gomoku9Logic()
        self.noise_scale = noise_scale

    def forward_for_mcts(self, batch):
        boards_flat = batch["boards"].float()
        current_player = batch["current_player"]
        B = boards_flat.shape[0]
        policies = torch.zeros(B, NA9)
        values = torch.zeros(B, 1)
        for b in range(B):
            board_np = boards_flat[b].numpy().reshape(BS9, BS9).astype(np.int8)
            player = int(current_player[b])
            logits = np.full(NA9, -1e6, dtype=np.float64)
            for r in range(BS9):
                for c in range(BS9):
                    idx = r * BS9 + c
                    if board_np[r, c] != 0:
                        continue
                    board_np[r, c] = player
                    if _check_win9(board_np, r, c, player):
                        logits[idx] = 100000.0
                    else:
                        logits[idx] = _eval_board9(board_np, player)
                    board_np[r, c] = 0
            valid = logits > -1e5
            if valid.any():
                noise = np.random.randn(NA9) * self.noise_scale
                std = max(1.0, logits[valid].std())
                logits[valid] += noise[valid] * std
            logits_t = torch.tensor(logits, dtype=torch.float32)
            policies[b] = torch.softmax(logits_t * 0.03, dim=-1)
            ev = _eval_board9(board_np, player)
            noise_v = np.random.randn() * 200.0
            values[b, 0] = float(np.clip(np.tanh((ev + noise_v) / 3000.0),
                                         -0.95, 0.95))
        return {"policy": policies, "value": values}


def _build_heatmap_positions_9x9(seed=42):
    """Build a few curated 9×9 mid-game positions for heatmap comparison."""
    positions = []

    # ── Position 1: Must block opponent's threat ────────────────────
    b = np.zeros((BS9, BS9), dtype=np.int8)
    b[4, 3] = 2; b[4, 4] = 2; b[4, 5] = 2          # White ○○○ horizontal
    b[3, 4] = 1; b[5, 3] = 1; b[5, 5] = 1          # Black nearby
    b[2, 3] = 1; b[6, 6] = 2                        # scattered
    positions.append(dict(
        board=b, player=1,
        name="Must block",
        desc="White has ○○○ — Black must block at (4,2) or (4,6) or lose",
    ))

    # ── Position 2: Open position, few stones ─────────────────────────
    b = np.zeros((BS9, BS9), dtype=np.int8)
    b[4, 4] = 1; b[3, 3] = 2; b[5, 5] = 1; b[3, 5] = 2
    b[6, 3] = 1; b[2, 6] = 2
    positions.append(dict(
        board=b, player=1,
        name="Scattered stones",
        desc="Open board — 75 empty cells, many legal moves",
    ))

    # ── Position 3: Threat situation ──────────────────────────────────
    b = np.zeros((BS9, BS9), dtype=np.int8)
    b[4, 3] = 1; b[4, 4] = 1; b[4, 5] = 1          # Black ●●● horizontal
    b[3, 4] = 2; b[5, 3] = 2; b[5, 5] = 2; b[3, 6] = 2  # White around
    b[2, 2] = 1; b[6, 6] = 1                        # distant Black
    positions.append(dict(
        board=b, player=1,
        name="3-in-a-row threat",
        desc="Black has ●●● — extend to 4 or 5 to win",
    ))

    return positions


def _draw_go_board_9(ax, board_size):
    """Draw a go-style board background for arbitrary size."""
    ax.set_xlim(-0.6, board_size - 0.4)
    ax.set_ylim(-0.6, board_size - 0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bg = mpatches.Rectangle(
        (-0.55, -0.55), board_size + 0.1, board_size + 0.1,
        facecolor=BOARD_COLOR, edgecolor="#8D6E63", linewidth=1.5, zorder=0)
    ax.add_patch(bg)
    for i in range(board_size):
        ax.plot([0, board_size - 1], [i, i], color=GRID_COLOR,
                linewidth=0.6, zorder=1)
        ax.plot([i, i], [0, board_size - 1], color=GRID_COLOR,
                linewidth=0.6, zorder=1)
    # Star points for 9×9
    if board_size == 9:
        for r, c in [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]:
            ax.plot(c, r, "o", color=GRID_COLOR, markersize=3.5, zorder=2)


def _draw_stones_n(ax, board, board_size):
    """Draw stones for arbitrary board size."""
    for r in range(board_size):
        for c in range(board_size):
            val = board[r, c]
            if val != 0:
                circle = mpatches.Circle(
                    (c, r), 0.38,
                    facecolor=PLAYER_COLORS[val],
                    edgecolor=PLAYER_EDGE[val],
                    linewidth=1.0, zorder=4)
                ax.add_patch(circle)


def _draw_heatmap_9(ax, board, visits, action_chosen, player, board_size, n_actions):
    """Draw board with visit-count heatmap for 9×9."""
    _draw_go_board_9(ax, board_size)
    _draw_stones_n(ax, board, board_size)

    total = visits.sum()
    visit_frac = visits / total if total > 0 else np.zeros(n_actions)
    cmap = plt.get_cmap(HEATMAP_CMAP)
    max_frac = max(visit_frac.max(), 1e-9)

    for idx in range(n_actions):
        if visits[idx] <= 0:
            continue
        r, c = divmod(idx, board_size)
        if board[r, c] != 0:
            continue
        intensity = visit_frac[idx] / max_frac
        color = cmap(intensity * 0.85)
        alpha = 0.25 + 0.65 * intensity
        circle = mpatches.Circle(
            (c, r), 0.36, facecolor=color, alpha=alpha,
            edgecolor="none", zorder=3)
        ax.add_patch(circle)

    # Mark chosen action — darkest red with white cross
    if action_chosen is not None:
        ar, ac = divmod(action_chosen, board_size)
        circle = mpatches.Circle(
            (ac, ar), 0.40, facecolor="#8B0000", alpha=0.92,
            edgecolor="#4A0000", linewidth=1.5, zorder=7)
        ax.add_patch(circle)
        ax.plot(ac, ar, marker="+", color="white",
                markersize=14, markeredgewidth=3.0, zorder=8)


def generate_heatmap_9x9(num_sims=512, seed=42, out_path=None,
                         model_type="random"):
    """Generate side-by-side visit-distribution heatmaps on 9×9 gomoku.

    Shows how PUCT spreads visits across many cells while Gumbel
    concentrates search on the most promising candidates via
    sequential halving.
    """
    logic = Gomoku9Logic()
    if model_type == "heuristic":
        model = HeuristicModel9(noise_scale=2.0)
        model_label = "Noisy Heuristic (threat-aware)"
    else:
        model = RandomModel9()
        model_label = "Random Model"
    model.eval()

    positions = _build_heatmap_positions_9x9(seed)
    algo_names = ["PUCT", "Gumbel Dense"]
    n_algos = len(algo_names)
    n_pos = len(positions)
    max_nodes = max(num_sims + 500, 5000)

    fig = plt.figure(figsize=(13, 6.5 * n_pos), facecolor="white")
    fig.suptitle(
        f"Visit Distribution Heatmap — 9×9 Gomoku, {model_label}, {num_sims} sims",
        fontsize=15, fontweight="bold", y=0.995, color="#333")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor="#222222", edgecolor="#000000",
                       label="Black stone"),
        mpatches.Patch(facecolor="#EEEEEE", edgecolor="#999999",
                       label="White stone"),
        mpatches.Patch(facecolor=plt.get_cmap(HEATMAP_CMAP)(0.6), alpha=0.65,
                       edgecolor="none",
                       label="Visit heatmap (darker = more visits)"),
        Line2D([0], [0], marker="+", color="white", markeredgecolor="#8B0000",
               markeredgewidth=3.0, markersize=10, markerfacecolor="#8B0000",
               linestyle="None", label="Algorithm's pick (dark red + cross)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=len(legend_elements),
               fontsize=8, frameon=True, fancybox=True, framealpha=0.85,
               edgecolor="#ccc", handlelength=1.5, handletextpad=0.4,
               columnspacing=1.2)

    # Layout: [label, board, bar] per position
    height_ratios = []
    for _i in range(n_pos):
        height_ratios.extend([0.5, 4, 1.2])
    gs = GridSpec(n_pos * 3, n_algos, figure=fig, hspace=0.25, wspace=0.20,
                  left=0.04, right=0.97, top=0.955, bottom=0.03,
                  height_ratios=height_ratios)

    for pos_idx, pos in enumerate(positions):
        board = pos["board"]
        player = pos["player"]
        empty = int((board == 0).sum())
        print(f"  [{pos_idx+1}/{n_pos}] {pos['name']}: {pos['desc']}")
        print(f"    Empty cells: {empty}")

        # ── Position header label ────────────────────────────────
        row_label = pos_idx * 3
        ax_label = fig.add_subplot(gs[row_label, :])
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.25,
            f"Position {pos_idx + 1}:  {pos['name']}  \u2014  {pos['desc']}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#444")
        if pos_idx > 0:
            ax_label.axhline(y=1.0, color="#BBBBBB", linewidth=1.2,
                             xmin=-0.02, xmax=1.02, clip_on=False)

        for col, algo_name in enumerate(algo_names):
            algo_seed = seed + pos_idx * 100 + col * 7
            np.random.seed(algo_seed)
            torch.manual_seed(algo_seed)

            if algo_name == "PUCT":
                tree = PUCT(n_games=1, max_nodes=max_nodes,
                            logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                tree.run_simulation_batch(
                    model, [0], num_simulations=num_sims)
                visits, root_q = tree.get_all_root_data(n_active=1)
                action = int(np.argmax(visits[0]))
            else:  # Gumbel Dense
                tree = GumbelDense(n_games=1, max_nodes=max_nodes,
                                   logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                move = tree.run_simulation_batch(
                    model, [0], num_simulations=num_sims)
                visits, root_q = tree.get_all_root_data(n_active=1)
                action = int(move[0])

            v = visits[0]
            q = float(root_q[0])
            spread = int((v > 0).sum())
            peak = int(v.max())
            ar, ac = divmod(action, BS9)

            print(f"    {algo_name:15s}  pick=({ar},{ac})  Q={q:+.3f}  "
                  f"peak={peak}  spread={spread}/{empty}")

            # ── Board heatmap ────────────────────────────────────────
            row_board = pos_idx * 3 + 1
            ax_board = fig.add_subplot(gs[row_board, col])
            _draw_heatmap_9(ax_board, board, v, action, player, BS9, NA9)
            ax_board.set_title(algo_name, fontsize=11, fontweight="bold",
                               color=ALGO_COLORS[algo_name], pad=5)

            # ── Bar chart (top-10 visited moves) ─────────────────────
            row_bar = pos_idx * 3 + 2
            ax_bar = fig.add_subplot(gs[row_bar, col])
            total_v = v.sum()
            if total_v > 0:
                k = 10
                top_idx = np.argsort(v)[::-1][:k]
                top_v = v[top_idx]
                pct = top_v / total_v * 100
                labels = [f"{i // BS9},{i % BS9}" for i in top_idx]
                bars = ax_bar.bar(range(len(top_idx)), pct,
                                 color=ALGO_COLORS[algo_name],
                                 alpha=0.7, edgecolor="white",
                                 linewidth=0.5)
                for bi, idx in enumerate(top_idx):
                    if idx == action:
                        bars[bi].set_alpha(1.0)
                        bars[bi].set_edgecolor("#333")
                        bars[bi].set_linewidth(1.5)
                ax_bar.set_xticks(range(len(top_idx)))
                ax_bar.set_xticklabels(labels, fontsize=6, rotation=45,
                                       ha="right")
                ax_bar.set_ylabel("visits %", fontsize=7)
                ax_bar.set_ylim(0, max(pct.max() * 1.3, 5))
            else:
                ax_bar.set_visible(False)
            ax_bar.tick_params(axis="y", labelsize=6)
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)



    out = out_path or "examples/gomoku_heatmap_9x9.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved heatmap → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visit-distribution heatmap: PUCT vs Gumbel on 9×9 Gomoku")
    parser.add_argument("--sims", type=int, default=64,
                        help="Simulations per position (default 64)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", type=str, default="heuristic",
                        choices=["random", "heuristic"],
                        help="Model type (default: heuristic)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file path (default: examples/gomoku_heatmap_9x9.png)")
    args = parser.parse_args()

    print(f"Generating 9×9 heatmap ({args.sims} sims, model={args.model}, "
          f"seed={args.seed})…")
    generate_heatmap_9x9(
        num_sims=args.sims, seed=args.seed, out_path=args.out,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
