"""
demo_puct_vs_gumbel.py — PUCT vs Gumbel Dense head-to-head on 15×15 Gomoku.

Plays full games where PUCT controls one side and Gumbel Dense the other
(alternating who goes first).  Sweeps the simulation budget from very low
(2) upward and plots PUCT win-rate vs budget.

Usage:
    uv run python examples/demo_puct_vs_gumbel.py                       # defaults
    uv run python examples/demo_puct_vs_gumbel.py --games 40 --seed 0   # more games
    uv run python examples/demo_puct_vs_gumbel.py --out winrate.png     # custom output
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_logic.gomoku import GomokuLogic
from gumbel_mcts import PUCT, GumbelDense
from numba import njit

# ── Constants ────────────────────────────────────────────────────────────────

BS = GomokuLogic.BOARD_SHAPE[0]  # 15
NA = GomokuLogic.NUM_ACTIONS     # 225


# ── Random model ─────────────────────────────────────────────────────────────


class RandomModel(nn.Module):
    """Near-uniform random policy for 15×15 Gomoku."""
    def __init__(self):
        super().__init__()
        self.logic = GomokuLogic()
        self.net = nn.Sequential(
            nn.Linear(NA, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, NA)
        self.value_head = nn.Linear(64, 1)

    def forward_for_mcts(self, batch):
        x = batch["boards"].float().reshape(-1, NA)
        h = self.net(x)
        policy = torch.softmax(self.policy_head(h) / 50.0, dim=-1)
        value = torch.tanh(self.value_head(h) * 0.01)
        return {"policy": policy, "value": value}


# ── Heuristic model (noisy prior) ───────────────────────────────────────────


@njit(cache=False)
def _eval_board(board, player):
    """Sliding-window heuristic evaluation for 15×15 (5-in-a-row)."""
    opponent = 3 - player
    score = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(BS):
            for c in range(BS):
                er = r + 4 * dr
                ec = c + 4 * dc
                if er < 0 or er >= BS or ec < 0 or ec >= BS:
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


@njit(cache=False)
def _check_win_15(board, r, c, player):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        total = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < BS and 0 <= cc < BS and board[rr, cc] == player:
            total += 1; rr += dr; cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < BS and 0 <= cc < BS and board[rr, cc] == player:
            total += 1; rr -= dr; cc -= dc
        if total >= 5:
            return True
    return False


class HeuristicModel(nn.Module):
    """Noisy threat-aware heuristic for 15×15 Gomoku.

    1-ply lookahead with Gaussian noise — informative but imperfect prior.
    """
    def __init__(self, noise_scale=2.0, temp=0.03, value_noise=200.0):
        super().__init__()
        self.logic = GomokuLogic()
        self.noise_scale = noise_scale
        self.temp = temp
        self.value_noise = value_noise

    def forward_for_mcts(self, batch):
        boards_flat = batch["boards"].float()
        current_player = batch["current_player"]
        B = boards_flat.shape[0]
        policies = torch.zeros(B, NA)
        values = torch.zeros(B, 1)
        for b in range(B):
            board_np = boards_flat[b].numpy().reshape(BS, BS).astype(np.int8)
            player = int(current_player[b])
            logits = np.full(NA, -1e6, dtype=np.float64)
            for r in range(BS):
                for c in range(BS):
                    idx = r * BS + c
                    if board_np[r, c] != 0:
                        continue
                    board_np[r, c] = player
                    if _check_win_15(board_np, r, c, player):
                        logits[idx] = 100000.0
                    else:
                        logits[idx] = _eval_board(board_np, player)
                    board_np[r, c] = 0
            valid = logits > -1e5
            if valid.any():
                noise = np.random.randn(NA) * self.noise_scale
                std = max(1.0, logits[valid].std())
                logits[valid] += noise[valid] * std
            logits_t = torch.tensor(logits, dtype=torch.float32)
            policies[b] = torch.softmax(logits_t * self.temp, dim=-1)
            ev = _eval_board(board_np, player)
            noise_v = np.random.randn() * self.value_noise
            values[b, 0] = float(np.clip(np.tanh((ev + noise_v) / 3000.0),
                                         -0.95, 0.95))
        return {"policy": policies, "value": values}


# ── Play one game ────────────────────────────────────────────────────────────

def play_game(puct_player, num_sims, model, logic, max_nodes, rng):
    """Play a full game: puct_player ∈ {1, 2} controls PUCT, the other uses Gumbel.

    Returns:
        "puct"   if PUCT's side wins,
        "gumbel" if Gumbel's side wins,
        "draw"   on a draw.
    """
    board = logic.get_initial_board()
    player = 1

    for _ in range(logic.MAX_MOVES):
        if player == puct_player:
            tree = PUCT(n_games=1, max_nodes=max_nodes,
                        logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([player]))
            tree.run_simulation_batch(model, [0], num_simulations=num_sims)
            visits, _ = tree.get_all_root_data(n_active=1)
            action = int(np.argmax(visits[0]))
        else:
            tree = GumbelDense(n_games=1, max_nodes=max_nodes,
                               logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([player]))
            move = tree.run_simulation_batch(model, [0],
                                             num_simulations=num_sims)
            action = int(move[0])

        _, winner, done, board = logic.fast_step(board, action, player)
        if done:
            if winner == puct_player:
                return "puct"
            elif winner == 0:
                return "draw"
            else:
                return "gumbel"
        player = 3 - player

    return "draw"


# ── Sweep budgets ────────────────────────────────────────────────────────────

def run_sweep(sim_budgets, games_per_budget, seed, model):
    logic = GomokuLogic()
    rng = np.random.RandomState(seed)

    results = {}
    for sims in sim_budgets:
        max_nodes = max(sims + 500, 5000)
        puct_wins = 0
        gumbel_wins = 0
        draws = 0
        t0 = time.time()

        for g in range(games_per_budget):
            # Alternate who goes first each game
            puct_player = 1 if g % 2 == 0 else 2
            game_seed = seed + sims * 1000 + g
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)

            outcome = play_game(puct_player, sims, model, logic, max_nodes, rng)
            if outcome == "puct":
                puct_wins += 1
            elif outcome == "gumbel":
                gumbel_wins += 1
            else:
                draws += 1

        elapsed = time.time() - t0
        total = puct_wins + gumbel_wins + draws
        puct_wr = puct_wins / total * 100
        gumbel_wr = gumbel_wins / total * 100
        draw_pct = draws / total * 100

        results[sims] = {
            "puct_wins": puct_wins,
            "gumbel_wins": gumbel_wins,
            "draws": draws,
            "puct_wr": puct_wr,
        }

        print(f"  sims={sims:4d}  |  PUCT {puct_wr:5.1f}%  "
              f"Gumbel {gumbel_wr:5.1f}%  Draw {draw_pct:5.1f}%  "
              f"({puct_wins}W/{gumbel_wins}L/{draws}D)  [{elapsed:.1f}s]")

    return results


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(results, games_per_budget, out_path, model_label="Random Model"):
    budgets = sorted(results.keys())
    puct_wr = [results[b]["puct_wr"] for b in budgets]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Win-rate line
    ax.plot(budgets, puct_wr, "o-", color="#5C6BC0", lw=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor="#5C6BC0",
            label="PUCT win-rate", zorder=5)

    # 50% reference
    ax.axhline(50, color="#BDBDBD", ls="--", lw=1, zorder=1)
    ax.text(budgets[-1], 51.5, "50%", ha="right", va="bottom",
            fontsize=8, color="#9E9E9E")

    # Labels showing Gumbel advantage below 50%
    for b, wr in zip(budgets, puct_wr):
        offset = 4 if wr < 50 else -4
        va = "top" if wr >= 50 else "bottom"
        ax.text(b, wr + offset, f"{wr:.0f}%", ha="center", va=va,
                fontsize=8, fontweight="bold", color="#5C6BC0")

    ax.set_xlabel("Simulation budget (both players)", fontsize=11)
    ax.set_ylabel("PUCT win-rate (%)", fontsize=11)
    ax.set_title(f"PUCT vs Gumbel Dense — 15×15 Gomoku ({model_label})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    # Shaded regions
    ax.axhspan(50, 105, alpha=0.04, color="#5C6BC0", zorder=0)
    ax.axhspan(-5, 50, alpha=0.04, color="#26A69A", zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    note = f"{games_per_budget} games per budget (alternating first player)"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#999")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PUCT vs Gumbel Dense win-rate sweep on 15×15 Gomoku")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per budget point (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output file path (auto-generated if omitted)")
    parser.add_argument("--budgets", type=str, default="2,4,8,16,32,64",
                        help="Comma-separated sim budgets (default: 2,4,8,16,32,64)")
    parser.add_argument("--model", type=str, default="random",
                        choices=["random", "heuristic", "strong"],
                        help="Model type (default: random)")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]

    if args.model == "strong":
        model_label = "Strong Heuristic"
        model = HeuristicModel(noise_scale=0.5, temp=0.06, value_noise=50.0)
    elif args.model == "heuristic":
        model_label = "Noisy Heuristic"
        model = HeuristicModel(noise_scale=2.0)
    else:
        model_label = "Random Model"
        model = RandomModel()

    out_path = args.out or f"examples/puct_vs_gumbel_winrate_{args.model}.png"

    print(f"PUCT vs Gumbel Dense — 15×15 Gomoku, {model_label}")
    print(f"Budgets: {budgets}")
    print(f"Games per budget: {args.games}")
    print(f"Seed: {args.seed}\n")

    torch.manual_seed(args.seed)
    model.eval()

    results = run_sweep(budgets, args.games, args.seed, model)
    plot_results(results, args.games, out_path, model_label=model_label)


if __name__ == "__main__":
    main()
