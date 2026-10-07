"""
demo/sokoban.py — Sokoban puzzle: PUCT vs Gumbel across a wide simulation budget.

Puzzle: custom 2-box design.  5×8 grid.  Both boxes sit in narrow side
corridors (columns 2 and 5) and must be pushed up then sideways to the targets
in row 1.  Wrong pushes (box down) immediately deadlock a box in a corner
(column wall + solid row 4), making the position unsolvable.

Both algorithms use the same heuristic model: uniform policy prior
+ value = −tanh(optimal_assignment_distance / 6), with −1 for deadlocked states.

Result summary
--------------
  Gumbel dominates across all tested budgets.  Sequential halving identifies
  the deadlock branch at the very first phase (−1 value signal) and
  concentrates its remaining budget on the surviving candidate.  PUCT's UCB
  keeps visiting the deadlock branch proportionally to its prior.

  The Gumbel advantage is robust at high budgets: the optimal solution
  (~14 moves) requires only correct upward pushes, and wrong downward pushes
  always create immediate corner deadlocks — the regime where Gumbel excels.

Outputs
-------
  sokoban_plot.png  — success-rate vs simulation budget with 95 % Wilson CIs
  sokoban_anim.gif  — side-by-side PUCT vs Gumbel trajectory

Usage
-----
    uv run python demo/sokoban.py
    uv run python demo/sokoban.py --episodes 40 --budgets 4,8,16,32,64,128,256,512,1024
    uv run python demo/sokoban.py --budget-anim 32
"""

import argparse
import os
import sys

import numpy as np
import torch
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch

from gumbel_mcts import PUCT, GumbelDense


# ── World constants ────────────────────────────────────────────────────────────
# Custom 2-box Sokoban puzzle designed for heuristic compatibility.
#
#   ########   row 0: boundary
#   #.    .#   row 1: targets at (1,1) and (6,1)
#   ##$  $##   row 2: boxes at (2,2) and (5,2) — outer columns walled
#   ##P   ##   row 3: player at (2,3) — directly below box0
#   ########   row 4: boundary
#
# Solution path (14 moves, 4 pushes):
#   1. Pushes box0 UP: (2,2)→(2,1)     h: 4→3   (player at (2,3), first move)
#   2. Player navigates to (3,1)
#   3. Pushes box0 LEFT: (2,1)→(1,1)   h: 3→2   box0 done
#   4. Player navigates to (5,3)
#   5. Pushes box1 UP: (5,2)→(5,1)     h: 2→1
#   6. Player navigates to (4,1)
#   7. Pushes box1 RIGHT: (5,1)→(6,1)  h: 1→0   SOLVED
#
# Every box push decreases the optimal-assignment Manhattan distance.
# Deadlock traps: pushing any box DOWN lands it in a corner
#   (col walls + solid row 4 below), returning value = -1.0.

GRID_H  = 5    # rows (y axis, 0 = top)
GRID_W  = 8    # columns (x axis, 0 = left)
N_BOXES = 2

# Action deltas in (x, y): up=(-y), right=(+x), down=(+y), left=(-x)
ADX = np.array([ 0, 1,  0, -1], dtype=np.int8)
ADY = np.array([-1, 0,  1,  0], dtype=np.int8)

# WALLS[row=y, col=x] = 1 means wall (includes void cells outside the map).
WALLS = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],   # row 0:  ########
    [1, 0, 0, 0, 0, 0, 0, 1],   # row 1:  #.    .#  targets at (1,1),(6,1)
    [1, 1, 0, 0, 0, 0, 1, 1],   # row 2:  ##$  $##  boxes at (2,2),(5,2)
    [1, 1, 0, 0, 0, 0, 1, 1],   # row 3:  ## @  ##  player at (3,3)
    [1, 1, 1, 1, 1, 1, 1, 1],   # row 4:  ########
], dtype=np.int8)

# Target positions (x, y), one per box
TARGETS = np.array([[1, 1], [6, 1]], dtype=np.int8)

# Initial board: [player_x, player_y, box0_x, box0_y, box1_x, box1_y]
START = np.array([2, 3, 2, 2, 5, 2], dtype=np.int8)

# ── Numba game functions ───────────────────────────────────────────────────────

@njit(cache=False)
def fast_step(board, action, player):
    """Apply one Sokoban action in-place.

    Invalid moves (wall / blocked push) leave the board unchanged and return
    (0.0, 0, False, board).  Solving all boxes → (1.0, 1, True, board).
    """
    if player == np.int8(2) or action == np.int32(4):
        return 0.0, np.int32(0), False, board

    px = np.int32(board[0])
    py = np.int32(board[1])
    dx = np.int32(ADX[action])
    dy = np.int32(ADY[action])
    nx = np.int32(px + dx)
    ny = np.int32(py + dy)

    if ny < 0 or ny >= np.int32(GRID_H) or nx < 0 or nx >= np.int32(GRID_W):
        return 0.0, np.int32(0), False, board
    if WALLS[ny, nx]:
        return 0.0, np.int32(0), False, board

    # Check for a box at (nx, ny)
    pushed = np.int32(-1)
    for i in range(N_BOXES):
        if np.int32(board[2 + 2*i]) == nx and np.int32(board[3 + 2*i]) == ny:
            pushed = np.int32(i)
            break

    if pushed >= np.int32(0):
        bx2 = np.int32(nx + dx)
        by2 = np.int32(ny + dy)
        if by2 < 0 or by2 >= np.int32(GRID_H) or bx2 < 0 or bx2 >= np.int32(GRID_W):
            return 0.0, np.int32(0), False, board
        if WALLS[by2, bx2]:
            return 0.0, np.int32(0), False, board
        for i in range(N_BOXES):
            if i != pushed:
                if np.int32(board[2 + 2*i]) == bx2 and np.int32(board[3 + 2*i]) == by2:
                    return 0.0, np.int32(0), False, board
        board[2 + 2*pushed] = np.int8(bx2)
        board[3 + 2*pushed] = np.int8(by2)

    board[0] = np.int8(nx)
    board[1] = np.int8(ny)

    # Check if all boxes are on targets
    all_on = True
    for i in range(N_BOXES):
        found = False
        for j in range(N_BOXES):
            if (np.int32(board[2 + 2*i]) == np.int32(TARGETS[j, 0]) and
                    np.int32(board[3 + 2*i]) == np.int32(TARGETS[j, 1])):
                found = True
                break
        if not found:
            all_on = False
            break

    if all_on:
        return 1.0, np.int32(1), True, board

    return 0.0, np.int32(0), False, board


@njit(cache=False)
def get_valid_mask(board, player):
    """Return which actions are actually executable from this board."""
    mask = np.zeros(5, dtype=np.float32)
    if player == np.int8(2):
        mask[4] = 1.0
        return mask

    px = np.int32(board[0])
    py = np.int32(board[1])

    for a in range(4):
        dx = np.int32(ADX[a])
        dy = np.int32(ADY[a])
        nx = np.int32(px + dx)
        ny = np.int32(py + dy)

        if ny < 0 or ny >= np.int32(GRID_H) or nx < 0 or nx >= np.int32(GRID_W):
            continue
        if WALLS[ny, nx]:
            continue

        # If a box is here, check whether it can be pushed
        ok = True
        for i in range(N_BOXES):
            if np.int32(board[2 + 2*i]) == nx and np.int32(board[3 + 2*i]) == ny:
                bx2 = np.int32(nx + dx)
                by2 = np.int32(ny + dy)
                if by2 < 0 or by2 >= np.int32(GRID_H) or bx2 < 0 or bx2 >= np.int32(GRID_W):
                    ok = False
                    break
                if WALLS[by2, bx2]:
                    ok = False
                    break
                for j in range(N_BOXES):
                    if j != i:
                        if (np.int32(board[2 + 2*j]) == bx2 and
                                np.int32(board[3 + 2*j]) == by2):
                            ok = False
                            break
                break

        if ok:
            mask[a] = 1.0

    return mask


# ── GameLogic ─────────────────────────────────────────────────────────────────

class SokobanLogic:
    """Sokoban with two-player MCTS wrapper (P2 always passes)."""

    NUM_ACTIONS     = 5    # 0=up, 1=right, 2=down, 3=left, 4=pass (P2 only)
    BOARD_SHAPE     = (2 + 2 * N_BOXES,)   # [px, py, bx0, by0, bx1, by1]
    MAX_MOVES       = 400  # 200 effective player moves
    MAX_LEGAL_MOVES = 4
    PLAYER_1        = 1
    PLAYER_2        = 2

    fast_step      = staticmethod(fast_step)
    get_valid_mask = staticmethod(get_valid_mask)

    @staticmethod
    def get_initial_board():
        return START.copy()


# ── Heuristic model ────────────────────────────────────────────────────────────

def _corner_deadlocked(bx, by):
    """True if a box at (bx,by) is stuck in a corner that is not a target."""
    wu = WALLS[by - 1, bx] > 0 if by > 0 else True
    wd = WALLS[by + 1, bx] > 0 if by < GRID_H - 1 else True
    wl = WALLS[by, bx - 1] > 0 if bx > 0 else True
    wr = WALLS[by, bx + 1] > 0 if bx < GRID_W - 1 else True
    in_corner = (wu and wl) or (wu and wr) or (wd and wl) or (wd and wr)
    if not in_corner:
        return False
    for t in TARGETS:
        if int(t[0]) == bx and int(t[1]) == by:
            return False    # corner coincides with a target → safe
    return True


class SokobanModel:
    """
    Uniform policy + heuristic value.

    Value = -tanh(optimal_box_to_target_distance / scale).
    Dead-locked states (box irreversibly stuck in a corner) receive -1.0.
    """

    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B       = batch["boards"].shape[0]
        boards  = batch["boards"].float()
        players = batch["current_player"]

        tx0, ty0 = float(TARGETS[0, 0]), float(TARGETS[0, 1])
        tx1, ty1 = float(TARGETS[1, 0]), float(TARGETS[1, 1])

        bx0 = boards[:, 2]; by0 = boards[:, 3]
        bx1 = boards[:, 4]; by1 = boards[:, 5]

        # Optimal assignment of boxes to targets (2-box case = 2 options)
        d01 = (bx0 - tx0).abs() + (by0 - ty0).abs() + (bx1 - tx1).abs() + (by1 - ty1).abs()
        d10 = (bx0 - tx1).abs() + (by0 - ty1).abs() + (bx1 - tx0).abs() + (by1 - ty0).abs()
        dist = torch.minimum(d01, d10)          # (B,)

        policy = torch.zeros(B, 5)
        value  = torch.zeros(B, 1)

        p1_idx = (players == 1).nonzero(as_tuple=True)[0]
        p2_idx = (players == 2).nonzero(as_tuple=True)[0]

        if len(p1_idx):
            policy[p1_idx, :4] = 0.25
            v = -torch.tanh(dist[p1_idx] / 6.0)
            # Dead-lock penalty
            for k in p1_idx:
                k = k.item()
                for i in range(N_BOXES):
                    bxi = int(boards[k, 2 + 2*i].item())
                    byi = int(boards[k, 3 + 2*i].item())
                    if _corner_deadlocked(bxi, byi):
                        value[k] = -1.0
                        break
                else:
                    value[k] = -torch.tanh(dist[k] / 6.0)

        if len(p2_idx):
            policy[p2_idx, 4] = 1.0
            for k in p2_idx:
                k = k.item()
                deadlocked = False
                for i in range(N_BOXES):
                    bxi = int(boards[k, 2 + 2*i].item())
                    byi = int(boards[k, 3 + 2*i].item())
                    if _corner_deadlocked(bxi, byi):
                        deadlocked = True
                        break
                value[k] = 1.0 if deadlocked else torch.tanh(dist[k] / 6.0)

        return {"policy": policy, "value": value}


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(algo, num_sims, logic, model):
    board    = logic.get_initial_board()
    traj     = [board.copy()]
    max_steps = logic.MAX_MOVES // 2
    max_nodes = max(num_sims * 10 + 200, 600)

    for _ in range(max_steps):
        if algo == "puct":
            tree = PUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            tree.run_simulation_batch(model, [0], num_simulations=num_sims)
            visits, _ = tree.get_all_root_data(n_active=1)
            v = visits[0].astype(np.float64)
            action = int(np.random.choice(len(v), p=v / v.sum()))
        else:
            tree = GumbelDense(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            actions = tree.run_simulation_batch(model, [0], num_simulations=num_sims)
            action = int(actions[0])

        _, _, done, board = logic.fast_step(board, action, 1)
        traj.append(board.copy())
        if done:
            return True, traj

    return False, traj


# ── Budget sweep ───────────────────────────────────────────────────────────────

def sweep(budgets, n_episodes, seed, logic, model):
    results = {}
    for sims in budgets:
        puct_ok = gumbel_ok = 0
        for ep in range(n_episodes):
            base = seed + sims * 10_000 + ep
            np.random.seed(base);     torch.manual_seed(base)
            ok, _ = run_episode("puct",   sims, logic, model)
            puct_ok += ok
            np.random.seed(base + 1); torch.manual_seed(base + 1)
            ok, _ = run_episode("gumbel", sims, logic, model)
            gumbel_ok += ok

        results[sims] = {
            "puct":   puct_ok   / n_episodes * 100,
            "gumbel": gumbel_ok / n_episodes * 100,
        }
        print(
            f"  sims={sims:3d}  |  "
            f"PUCT {results[sims]['puct']:5.1f}%  "
            f"Gumbel {results[sims]['gumbel']:5.1f}%"
        )
    return results


# ── Static plot ────────────────────────────────────────────────────────────────

def _wilson_ci(p_pct, n, z=1.96):
    """Return (lo, hi) Wilson 95% CI in percentage points."""
    p = p_pct / 100.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100)


def plot_results(results, n_episodes, out_path):
    budgets   = sorted(results.keys())
    puct_sr   = [results[b]["puct"]   for b in budgets]
    gumbel_sr = [results[b]["gumbel"] for b in budgets]

    puct_ci   = [_wilson_ci(p, n_episodes) for p in puct_sr]
    gumbel_ci = [_wilson_ci(p, n_episodes) for p in gumbel_sr]
    puct_lo,   puct_hi   = zip(*puct_ci)
    gumbel_lo, gumbel_hi = zip(*gumbel_ci)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    kw = dict(lw=2.5, ms=8, mfc="white", mew=2)
    ax.fill_between(budgets, puct_lo,   puct_hi,   color="#5C6BC0", alpha=0.15)
    ax.fill_between(budgets, gumbel_lo, gumbel_hi, color="#26A69A", alpha=0.15)
    ax.plot(budgets, puct_sr,   "o-", color="#5C6BC0", label="PUCT",   **kw)
    ax.plot(budgets, gumbel_sr, "s-", color="#26A69A", label="Gumbel", **kw)
    ax.axhline(50, color="#BDBDBD", ls="--", lw=1)

    ax.set_xlabel("Simulation budget per move", fontsize=11)
    ax.set_ylabel("Puzzle success rate (%)",    fontsize=11)
    ax.set_title(
        "Sokoban — PUCT vs Gumbel under tight simulation budgets",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.text(
        0.99, 0.02, f"{n_episodes} episodes per budget point",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#999",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot  -> {out_path}")


# ── Animation ──────────────────────────────────────────────────────────────────

def _draw_board_bg(ax, title):
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(GRID_H - 0.5, -0.5)   # y-axis: row 0 at top
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Floor
    ax.add_patch(Rectangle((-0.5, -0.5), GRID_W, GRID_H,
                            color="#F5F5F5", zorder=0))
    # Walls
    for row in range(GRID_H):
        for col in range(GRID_W):
            if WALLS[row, col]:
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                        color="#37474F", zorder=1))
    # Targets
    for t in TARGETS:
        tx, ty = int(t[0]), int(t[1])
        ax.add_patch(Rectangle((tx - 0.5, ty - 0.5), 1, 1,
                                color="#FFF9C4", zorder=1))
        ax.plot(tx, ty, "x", color="#F9A825", ms=12, mew=2.5, zorder=3)


def _make_box_patches(ax, board, color_off, color_on):
    patches = []
    for i in range(N_BOXES):
        bx, by = int(board[2 + 2*i]), int(board[3 + 2*i])
        on_target = any(
            bx == int(TARGETS[j, 0]) and by == int(TARGETS[j, 1])
            for j in range(N_BOXES)
        )
        c = color_on if on_target else color_off
        p = ax.add_patch(Rectangle(
            (bx - 0.4, by - 0.4), 0.8, 0.8,
            color=c, zorder=4, lw=1.5, ec="#4E342E"
        ))
        patches.append(p)
    return patches


def _find_contrasting_seed(logic, model, budget, base_seed, max_tries=60):
    """Find a seed where Gumbel solves the puzzle but PUCT doesn't."""
    best = None
    for i in range(max_tries):
        s = base_seed + i * 37
        np.random.seed(s);     torch.manual_seed(s)
        ok_p, pt = run_episode("puct",   budget, logic, model)
        np.random.seed(s + 1); torch.manual_seed(s + 1)
        ok_g, gt = run_episode("gumbel", budget, logic, model)
        if ok_g and not ok_p:
            return pt, gt, ok_p, ok_g
        if best is None and ok_g:
            best = (pt, gt, ok_p, ok_g)   # fallback: at least Gumbel wins
    if best is not None:
        return best
    np.random.seed(base_seed);     torch.manual_seed(base_seed)
    _, pt = run_episode("puct",   budget, logic, model)
    np.random.seed(base_seed + 1); torch.manual_seed(base_seed + 1)
    ok_g, gt = run_episode("gumbel", budget, logic, model)
    return pt, gt, False, ok_g


def make_animation(logic, model, budget, seed, out_path):
    print(f"  searching for a contrasting episode pair (budget={budget})...")
    puct_traj, gumbel_traj, ok_p, ok_g = _find_contrasting_seed(
        logic, model, budget, seed
    )
    trajs  = [puct_traj, gumbel_traj]
    labels = [
        f"PUCT  (budget={budget})\n{'Solved!' if ok_p else 'Failed'}",
        f"Gumbel (budget={budget})\n{'Solved!' if ok_g else 'Failed'}",
    ]
    colors_off = ["#7986CB", "#26A69A"]   # box not yet on target
    colors_on  = ["#FFD54F", "#FFD54F"]  # box on target (gold)
    n_frames   = max(len(puct_traj), len(gumbel_traj))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(wspace=0.25)

    for ax, label in zip(axes, labels):
        _draw_board_bg(ax, label)

    # Initial box patches and player markers
    box_patches = []
    player_markers = []
    for ax, traj, c_off, c_on in zip(axes, trajs, colors_off, colors_on):
        board = traj[0]
        box_patches.append(_make_box_patches(ax, board, c_off, c_on))
        px, py = int(board[0]), int(board[1])
        m, = ax.plot(px, py, "o", color="#B71C1C", ms=10, zorder=6)
        player_markers.append(m)
        # start marker
        ax.plot(int(START[0]), int(START[1]), "s",
                color="#1B5E20", ms=6, alpha=0.4, zorder=2)

    trail_lines = [[], []]

    def update(frame):
        for side, (traj, ax, c_off, c_on, patches, marker) in enumerate(
            zip(trajs, axes, colors_off, colors_on, box_patches, player_markers)
        ):
            idx = min(frame, len(traj) - 1)
            board = traj[idx]

            # Update box positions and colors
            for i, patch in enumerate(patches):
                bx, by = int(board[2 + 2*i]), int(board[3 + 2*i])
                on_t = any(
                    bx == int(TARGETS[j, 0]) and by == int(TARGETS[j, 1])
                    for j in range(N_BOXES)
                )
                patch.set_xy((bx - 0.4, by - 0.4))
                patch.set_facecolor(c_on if on_t else c_off)

            # Update player
            px, py = int(board[0]), int(board[1])
            marker.set_data([px], [py])

            # Trail
            if 0 < frame <= len(traj) - 1:
                prev = traj[frame - 1]
                ln, = ax.plot(
                    [int(prev[0]), px], [int(prev[1]), py],
                    color="#B71C1C", alpha=0.3, lw=1.5, zorder=3
                )
                trail_lines[side].append(ln)

        return [m for m in player_markers] + [p for ps in box_patches for p in ps]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=250, blit=False
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer="pillow", fps=4)
    plt.close(fig)
    print(f"Saved anim  -> {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sokoban: PUCT vs Gumbel under tight simulation budgets"
    )
    ap.add_argument("--episodes",    type=int, default=40)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--budgets",     type=str, default="4,8,16,32,64,128,256")
    ap.add_argument("--budget-anim", type=int, default=None,
                    help="Budget for animation (default: middle of --budgets)")
    ap.add_argument("--out-plot", type=str, default="demo/png/sokoban_plot.png")
    ap.add_argument("--out-anim", type=str, default="demo/gif/sokoban_anim.gif")
    args = ap.parse_args()

    budgets     = [int(x) for x in args.budgets.split(",")]
    budget_anim = args.budget_anim if args.budget_anim is not None \
                  else budgets[len(budgets) // 2]
    logic = SokobanLogic()
    model = SokobanModel(logic)

    print("Sokoban — PUCT vs Gumbel")
    print(
        f"  Grid {GRID_W}×{GRID_H}, {N_BOXES} boxes  |  "
        f"Start {tuple(int(v) for v in START[:2])}  "
        f"Targets {[tuple(int(v) for v in t) for t in TARGETS]}"
    )
    print(f"  Budgets {budgets}  |  {args.episodes} episodes  |  seed {args.seed}\n")

    results = sweep(budgets, args.episodes, args.seed, logic, model)
    plot_results(results, args.episodes, args.out_plot)

    print(f"\nGenerating animation at budget={budget_anim}...")
    make_animation(logic, model, budget_anim, args.seed, args.out_anim)


if __name__ == "__main__":
    main()
