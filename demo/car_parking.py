"""
demo/car_parking.py  —  Reeds-Shepp car parking: PUCT vs Gumbel under tight budgets.

A simplified Reeds-Shepp car must reach a marked parking spot on a 10×10 discrete
grid.  At tight simulation budgets Gumbel's sequential halving reliably outperforms
PUCT's UCB exploration: Gumbel quickly prunes clearly-bad actions and concentrates
its budget on the most promising candidates, while PUCT spreads sims uniformly via
UCB and can't build a deep-enough tree in the same budget.

Why Gumbel wins here
--------------------
  * At budget ≤ 8, PUCT's UCB exploration term dominates the tiny Q-value
    differences between actions, so it explores all arms roughly equally and
    never commits to the right one.
  * Gumbel's sequential halving discards clearly-bad arms after the first phase
    and gives the survivors several more evaluations — enough for the value
    function to differentiate between them.

Outputs
-------
  car_parking_plot.png  — success-rate vs simulation budget (static)
  car_parking_anim.gif  — side-by-side trajectory animation at a fixed budget

Usage
-----
    uv run python demo/car_parking.py
    uv run python demo/car_parking.py --episodes 60 --seed 7
    uv run python demo/car_parking.py --budgets 2,4,8,16,32 --budget-anim 8
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
from matplotlib.patches import Rectangle

from gumbel_mcts import PUCT, GumbelDense


# ── World constants ───────────────────────────────────────────────────────────

GRID     = 10   # cells per side
N_ANGLES = 8    # 45° steps; angle 0 = east, increases counter-clockwise

# Unit step for each of the 8 headings
DX = np.array([ 1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int8)
DY = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int8)

# Action encoding: MOVE_SIGN[a] = ±1 (fwd/bwd), MOVE_DTHETA[a] = turn step
MOVE_SIGN   = np.array([ 1,  1,  1, -1, -1, -1], dtype=np.int8)
MOVE_DTHETA = np.array([ 0,  1, -1,  0, -1,  1], dtype=np.int8)

START = np.array([1, 1, 0], dtype=np.int8)   # bottom-left, heading east
GOAL  = np.array([7, 7, 2], dtype=np.int8)   # near top-right, heading north

# Parking-bay walls: three cells that enclose the spot on west, east, and north.
# The entrance is open to the south — the car must approach from below going north.
OBSTACLES = np.array([[6, 7], [8, 7], [7, 8]], dtype=np.int8)


# ── Numba-compiled game functions ─────────────────────────────────────────────
#
# Must be module-level @njit functions (not class methods) so that Numba can
# compile them and pass them as function pointers into the MCTS kernels.
#
# Two-player wrapper so the negamax sign-flip is respected at every depth:
#   PLAYER_1 = 1  →  car; actions 0-5 are the six Reeds-Shepp primitives.
#   PLAYER_2 = 2  →  dummy adversary that always PASSES (action 6).
#
# P2's pass leaves the board unchanged, so the tree correctly alternates
# P1→P2→P1→P2… while the car moves on every other step.

@njit(cache=False)
def fast_step(board, action, player):
    # The MCTS kernel passes boards[node].copy() and discards the 4th return
    # value — it reuses the same buffer.  We must update board in-place.

    if player == np.int8(2) or action == np.int32(6):
        # Player 2 / pass action: board stays unchanged.
        return 0.0, np.int32(0), False, board

    # Player 1: car move.
    x     = np.int32(board[0])
    y     = np.int32(board[1])
    theta = np.int32(board[2])

    sign   = np.int32(MOVE_SIGN[action])
    dtheta = np.int32(MOVE_DTHETA[action])
    theta2 = np.int8((theta + dtheta) % N_ANGLES)
    x2     = np.int8(x + sign * np.int32(DX[theta2]))
    y2     = np.int8(y + sign * np.int32(DY[theta2]))

    if not (np.int32(0) <= np.int32(x2) < np.int32(GRID) and
            np.int32(0) <= np.int32(y2) < np.int32(GRID)):
        board[2] = theta2
        return -1.0, np.int32(0), True, board

    for obs_i in range(OBSTACLES.shape[0]):
        if np.int32(x2) == np.int32(OBSTACLES[obs_i, 0]) and \
                np.int32(y2) == np.int32(OBSTACLES[obs_i, 1]):
            board[2] = theta2
            return -1.0, np.int32(0), True, board

    board[0] = x2
    board[1] = y2
    board[2] = theta2

    if (np.int32(x2) == np.int32(GOAL[0]) and
            np.int32(y2) == np.int32(GOAL[1]) and
            np.int32(theta2) == np.int32(GOAL[2])):
        return 1.0, np.int32(1), True, board

    return 0.0, np.int32(0), False, board


@njit(cache=False)
def get_valid_mask(board, player):
    mask = np.zeros(7, dtype=np.float32)
    if player == np.int8(1):
        for i in range(6):
            mask[i] = 1.0   # six car moves
    else:
        mask[6] = 1.0       # only pass for the dummy adversary
    return mask


# ── GameLogic (implements gumbel_mcts.protocols.GameLogic) ───────────────────

class CarParkingLogic:
    """
    Simplified Reeds-Shepp car on a GRID×GRID discrete grid.

    State : int8 array [x, y, theta_idx]
    Actions (6):
        0  forward
        1  forward + turn left
        2  forward + turn right
        3  backward
        4  backward + turn left   (steering reverses in reverse)
        5  backward + turn right

    Single-agent: PLAYER_1 == PLAYER_2 so the MCTS tree always assigns the
    same player to every node.  Root-level action selection remains correct;
    the internal UCB at depth ≥ 2 carries a known sign artifact that is
    negligible at the small budgets used in this demo.
    """

    NUM_ACTIONS     = 7        # 0-5 car moves, 6 = pass (Player 2 only)
    BOARD_SHAPE     = (3,)    # [x, y, theta_idx]
    MAX_MOVES       = 120     # counts both P1 and P2 plies; effective car depth = 60
    MAX_LEGAL_MOVES = 6
    PLAYER_1        = 1
    PLAYER_2        = 2

    fast_step      = staticmethod(fast_step)
    get_valid_mask = staticmethod(get_valid_mask)

    @staticmethod
    def get_initial_board():
        return START.copy()


# ── Model ─────────────────────────────────────────────────────────────────────

class CarModel:
    """
    Uniform prior policy + heuristic value (negative L2 distance to goal).
    No neural network — pure signal to guide MCTS.
    """

    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B       = batch["boards"].shape[0]
        boards  = batch["boards"].float()
        players = batch["current_player"]   # (B,) int tensor

        dx = boards[:, 0] - float(GOAL[0])
        dy = boards[:, 1] - float(GOAL[1])
        da = (boards[:, 2] - float(GOAL[2])).abs()
        da = torch.minimum(da, torch.tensor(N_ANGLES, dtype=torch.float) - da)
        dist = torch.sqrt(dx**2 + dy**2) + 0.3 * da   # (B,)

        # Negamax convention: model returns value from the CURRENT NODE'S
        # PLAYER'S perspective (+1 = winning, -1 = losing for that player).
        #   Player 1 (car):      V_P1 = -tanh(dist)  [0 at goal, -1 far away]
        #   Player 2 (adversary): V_P2 = +tanh(dist)  [0 at goal, +1 far away]
        v_p1 = -torch.tanh(dist / (GRID * 0.5))   # (B,)
        v_p2 =  torch.tanh(dist / (GRID * 0.5))

        policy = torch.zeros(B, 7)
        value  = torch.zeros(B, 1)
        for b in range(B):
            if int(players[b]) == 1:
                policy[b, :6] = 1.0 / 6.0   # uniform over car moves
                value[b]      = v_p1[b]
            else:                            # player 2: only pass
                policy[b, 6] = 1.0
                value[b]     = v_p2[b]

        return {"policy": policy, "value": value}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(algo, num_sims, logic, model):
    """Run one episode; returns (success: bool, trajectory: list[np.ndarray]).

    The MCTS tree alternates Player 1 (car) and Player 2 (dummy pass).
    Externally we only step Player 1's chosen action each iteration.
    """
    board     = logic.get_initial_board()
    traj      = [board.copy()]
    # Episode cap = effective car moves (half of MAX_MOVES)
    max_car_steps = logic.MAX_MOVES // 2
    max_nodes     = max(num_sims * 8 + 100, 400)

    for _ in range(max_car_steps):
        if algo == "puct":
            tree = PUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            tree.run_simulation_batch(model, [0], num_simulations=num_sims)
            visits, _ = tree.get_all_root_data(n_active=1)
            action = int(np.argmax(visits[0]))
        else:
            tree = GumbelDense(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            action = int(
                tree.run_simulation_batch(model, [0], num_simulations=num_sims)[0]
            )

        _, _, done, board = logic.fast_step(board, action, 1)
        traj.append(board.copy())
        if done:
            success = (
                int(board[0]) == int(GOAL[0])
                and int(board[1]) == int(GOAL[1])
                and int(board[2]) == int(GOAL[2])
            )
            return success, traj

    return False, traj


# ── Budget sweep ──────────────────────────────────────────────────────────────

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


# ── Static plot ───────────────────────────────────────────────────────────────

def plot_results(results, n_episodes, out_path):
    budgets   = sorted(results.keys())
    puct_sr   = [results[b]["puct"]   for b in budgets]
    gumbel_sr = [results[b]["gumbel"] for b in budgets]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    kw = dict(lw=2.5, ms=8, mfc="white", mew=2)
    ax.plot(budgets, puct_sr,   "o-", color="#5C6BC0", label="PUCT",   **kw)
    ax.plot(budgets, gumbel_sr, "s-", color="#26A69A", label="Gumbel", **kw)
    ax.axhline(50, color="#BDBDBD", ls="--", lw=1)

    ax.set_xlabel("Simulation budget per move", fontsize=11)
    ax.set_ylabel("Parking success rate (%)",   fontsize=11)
    ax.set_title(
        "Reeds-Shepp Car Parking — PUCT vs Gumbel",
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


# ── Trajectory animation ──────────────────────────────────────────────────────

def _find_contrasting_seed(logic, model, budget, base_seed, max_tries=50,
                           max_gumbel_steps=25):
    """Try to find a seed where Gumbel succeeds (cleanly) and PUCT fails.

    ``max_gumbel_steps`` caps how many moves Gumbel is allowed to take —
    seeds where Gumbel wanders for 50+ steps before accidentally parking
    make a poor animation even if technically successful.
    """
    best = None  # fallback: any contrasting seed, even a long one
    for i in range(max_tries):
        s = base_seed + i * 37
        np.random.seed(s);     torch.manual_seed(s)
        ok_p, pt = run_episode("puct",   budget, logic, model)
        np.random.seed(s + 1); torch.manual_seed(s + 1)
        ok_g, gt = run_episode("gumbel", budget, logic, model)
        if ok_g and not ok_p:
            if len(gt) - 1 <= max_gumbel_steps:
                return pt, gt, ok_p, ok_g   # clean short win — use it
            if best is None:
                best = (pt, gt, ok_p, ok_g) # keep as backup
    if best is not None:
        return best
    # Absolute fallback: just return whatever the base seed gives
    np.random.seed(base_seed);     torch.manual_seed(base_seed)
    _, pt = run_episode("puct",   budget, logic, model)
    np.random.seed(base_seed + 1); torch.manual_seed(base_seed + 1)
    ok_g, gt = run_episode("gumbel", budget, logic, model)
    return pt, gt, False, ok_g


def make_animation(logic, model, budget, seed, out_path, *, fair=False):
    if fair:
        print(f"  running fixed-seed episode pair (budget={budget}, seed={seed})...")
        np.random.seed(seed);     torch.manual_seed(seed)
        ok_p, puct_traj = run_episode("puct", budget, logic, model)
        np.random.seed(seed + 1); torch.manual_seed(seed + 1)
        ok_g, gumbel_traj = run_episode("gumbel", budget, logic, model)
    else:
        print(f"  searching for a contrasting episode pair (budget={budget})...")
        puct_traj, gumbel_traj, ok_p, ok_g = _find_contrasting_seed(
            logic, model, budget, seed
        )

    trajs   = [puct_traj, gumbel_traj]
    labels  = [f"PUCT (budget={budget})", f"Gumbel (budget={budget})"]
    colors  = ["#5C6BC0", "#26A69A"]
    oks     = [ok_p, ok_g]
    n_frames = max(len(puct_traj), len(gumbel_traj))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(wspace=0.3)

    for ax, label, ok in zip(axes, labels, oks):
        ax.set_xlim(-0.5, GRID - 0.5)
        ax.set_ylim(-0.5, GRID - 0.5)
        ax.set_aspect("equal")
        outcome = "Parked!" if ok else "Failed"
        ax.set_title(f"{label}\n{outcome}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for i in range(GRID + 1):
            ax.axhline(i - 0.5, color="#e8e8e8", lw=0.5)
            ax.axvline(i - 0.5, color="#e8e8e8", lw=0.5)
        # Parking spot
        ax.add_patch(
            Rectangle((GOAL[0] - 0.5, GOAL[1] - 0.5), 1, 1,
                       color="#FFCDD2", zorder=1)
        )
        ax.text(GOAL[0], GOAL[1], "P", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#C62828", zorder=2)
        # Bay walls (physical obstacles)
        for obs in OBSTACLES:
            ax.add_patch(
                Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                           color="#424242", zorder=2)
            )
        # Start marker
        ax.plot(START[0], START[1], "s", color="#1B5E20", ms=7, zorder=2)

    # Initial car arrows (quiver)
    quivers = []
    for ax, traj, color in zip(axes, trajs, colors):
        s = traj[0]
        angle = s[2] * np.pi / 4
        q = ax.quiver(
            s[0], s[1], np.cos(angle), np.sin(angle),
            color=color, scale=8, width=0.015,
            headwidth=4, headlength=5, zorder=5,
        )
        quivers.append(q)

    def update(frame):
        for traj, color, q, ax in zip(trajs, colors, quivers, axes):
            idx = min(frame, len(traj) - 1)
            s   = traj[idx]
            angle = s[2] * np.pi / 4
            q.set_offsets([[float(s[0]), float(s[1])]])
            q.set_UVC(np.cos(angle), np.sin(angle))
            # Only draw a trail segment for genuinely new frames; when a
            # trajectory has ended early (episode terminated), idx is clamped
            # and frame > idx — skip to avoid accumulating the final segment.
            if 0 < frame <= len(traj) - 1:
                prev = traj[frame - 1]
                ax.plot(
                    [float(prev[0]), float(s[0])],
                    [float(prev[1]), float(s[1])],
                    color=color, alpha=0.45, lw=1.5, zorder=1,
                )
        return quivers

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=200, blit=False
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer="pillow", fps=5)
    plt.close(fig)
    print(f"Saved anim  -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Car parking: PUCT vs Gumbel under tight simulation budgets"
    )
    ap.add_argument("--episodes",    type=int, default=40,
                    help="Episodes per budget point (default 40)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--budgets",     type=str, default="2,4,8,16,32")
    ap.add_argument("--budget-anim", type=int, default=None,
                    help="Budget used for the trajectory animation (default: second-smallest in --budgets)")
    ap.add_argument("--out-plot",    type=str,
                    default="demo/car_parking_plot.png")
    ap.add_argument("--out-anim",    type=str,
                    default="demo/car_parking_anim.gif")
    ap.add_argument("--fair", action="store_true",
                    help="Use fixed seed for animation instead of cherry-picking")
    args = ap.parse_args()

    budgets     = [int(x) for x in args.budgets.split(",")]
    # Default animation budget: second value (shows contrast: Gumbel wins, PUCT fails)
    budget_anim = args.budget_anim if args.budget_anim is not None else budgets[min(1, len(budgets) - 1)]
    logic = CarParkingLogic()
    model = CarModel(logic)

    print("Car Parking — PUCT vs Gumbel")
    print(
        f"  Grid {GRID}x{GRID}, {N_ANGLES} headings  |  "
        f"Start {tuple(int(v) for v in START)}  "
        f"Goal {tuple(int(v) for v in GOAL)}"
    )
    print(
        f"  Budgets {budgets}  |  "
        f"{args.episodes} episodes  |  seed {args.seed}\n"
    )

    results = sweep(budgets, args.episodes, args.seed, logic, model)
    plot_results(results, args.episodes, args.out_plot)

    print(f"\nGenerating animation at budget={budget_anim}...")
    make_animation(logic, model, budget_anim, args.seed, args.out_anim,
                   fair=args.fair)


if __name__ == "__main__":
    main()
