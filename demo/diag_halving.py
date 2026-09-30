"""
diag_halving.py — Probe Gumbel sequential halving quality at sims=32 vs sims=64.

For each sim budget, runs N self-play episodes and records per-move:
  - k_initial, num_phases, sims_per_action in phase 0
  - improved policy entropy and max probability
  - visit count entropy (how concentrated the visits are)
  - fraction of k_initial candidates that received >1 sim (exploration coverage)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

from gumbel_mcts import GumbelDense

# ── Gomoku 9x9 logic (same as training) ───────────────────────────────────────
BS = 9
NA = BS * BS

@njit(cache=False)
def _check_win(board, r, c, player):
    for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
        n = 1
        for s in (1,-1):
            rr, cc = r+s*dr, c+s*dc
            while 0<=rr<BS and 0<=cc<BS and board[rr,cc]==player:
                n+=1; rr+=s*dr; cc+=s*dc
        if n>=5: return True
    return False

@njit(cache=False)
def _fast_step(board, action, player):
    r, c = action//BS, action%BS
    board[r,c] = player
    if _check_win(board,r,c,player): return 1.0, player, True, board
    for i in range(BS):
        for j in range(BS):
            if board[i,j]==0: return 0.0, 0, False, board
    return 0.0, 0, True, board

@njit(cache=False)
def _valid_mask(board, player):
    m = np.zeros(NA, dtype=np.float32)
    for r in range(BS):
        for c in range(BS):
            if board[r,c]==0: m[r*BS+c]=1.0
    return m

class Gomoku9Logic:
    NUM_ACTIONS = NA; BOARD_SHAPE = (BS,BS); MAX_MOVES = NA
    PLAYER_1 = 1; PLAYER_2 = 2
    fast_step = staticmethod(_fast_step)
    get_valid_mask = staticmethod(_valid_mask)

# ── Tiny model (random-ish, like early training) ───────────────────────────────
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NA, 64)
        self.ph = nn.Linear(64, NA)
        self.vh = nn.Linear(64, 1)
        self.logic = Gomoku9Logic()

    def forward_for_mcts(self, batch):
        x = batch["boards"].float().view(batch["boards"].shape[0], -1)
        h = torch.relu(self.fc(x))
        pol = F.softmax(self.ph(h) / 10.0, dim=-1)   # soft but not flat
        val = torch.tanh(self.vh(h) * 0.1)
        return {"policy": pol, "value": val.squeeze(-1)}

# ── Halving parameter calculator (mirrors gumbel_dense.py logic) ──────────────
def halving_params(sims, max_k=16, num_actions=NA):
    max_k = min(num_actions, max_k)
    num_phases = max(1, int(np.log2(max_k)))
    first_phase_budget = sims // num_phases
    k_initial = min(max_k, first_phase_budget // 2)
    k_initial = max(2, k_initial)
    num_phases = max(1, int(np.log2(k_initial)))
    sims_per_action_phase0 = (sims // num_phases) // k_initial
    return k_initial, num_phases, sims_per_action_phase0

# ── Diagnostic collection ──────────────────────────────────────────────────────
def collect_diagnostics(sims, n_episodes, logic, model, seed=42, max_k=16):
    np.random.seed(seed); torch.manual_seed(seed)
    k_initial, num_phases, spa0 = halving_params(sims, max_k)

    records = []   # one entry per move
    for ep in range(n_episodes):
        board  = np.zeros((BS,BS), dtype=np.int8)
        player = 1
        for _ in range(logic.MAX_MOVES):
            tree = GumbelDense(
                n_games=1, max_nodes=max(sims*5+200, 1000),
                logic=logic, device="cpu",
                max_considered_actions=max_k,
                entropy_min_k=max_k,   # disable entropy scheduler
            )
            tree.initialize_roots([0], board[None], np.array([player]))
            with torch.no_grad():
                tree.run_simulation_batch(model, [0], num_simulations=sims)

            # Improved policy
            imp_pol = tree.get_improved_policy(n_active=1)[0]
            imp_pol = np.clip(imp_pol, 1e-10, 1.0)
            imp_pol /= imp_pol.sum()
            imp_ent  = float(-np.sum(imp_pol * np.log(imp_pol)))
            imp_max  = float(imp_pol.max())

            # Visit distribution
            visits, _ = tree.get_all_root_data(n_active=1)
            v = visits[0]
            n_visited = int((v > 0).sum())
            if v.sum() > 0:
                vp = v / v.sum()
                vis_ent = float(-np.sum(vp[vp>0] * np.log(vp[vp>0])))
            else:
                vis_ent = 0.0

            records.append({
                "imp_ent":  imp_ent,
                "imp_max":  imp_max,
                "vis_ent":  vis_ent,
                "n_visited": n_visited,
            })

            # Play the move
            action = int(np.argmax(imp_pol))
            _, winner, done, board = logic.fast_step(board, action, player)
            if done: break
            player = 3 - player

        if (ep+1) % 20 == 0:
            print(f"  sims={sims}  ep={ep+1}/{n_episodes}  "
                  f"k_initial={k_initial}  phases={num_phases}  spa0={spa0}")
    return records, k_initial, num_phases, spa0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=str, default="demo/diag_halving.png")
    args = ap.parse_args()

    logic = Gomoku9Logic()
    # Warm up Numba
    dummy = np.zeros((BS,BS), dtype=np.int8)
    _fast_step(dummy, 0, 1); _valid_mask(dummy, 1)

    model = TinyNet(); model.eval()

    budgets = [32, 64]
    all_records = {}
    all_params  = {}

    for sims in budgets:
        print(f"\n--- Collecting sims={sims} ---")
        recs, k_init, nph, spa0 = collect_diagnostics(
            sims, args.episodes, logic, model, seed=42
        )
        all_records[sims] = recs
        all_params[sims]  = dict(k_initial=k_init, num_phases=nph, spa0=spa0)
        print(f"  k_initial={k_init}  num_phases={nph}  sims_per_action_phase0={spa0}")
        print(f"  imp_ent  mean={np.mean([r['imp_ent']  for r in recs]):.3f}  "
              f"std={np.std([r['imp_ent']  for r in recs]):.3f}")
        print(f"  imp_max  mean={np.mean([r['imp_max']  for r in recs]):.3f}  "
              f"std={np.std([r['imp_max']  for r in recs]):.3f}")
        print(f"  vis_ent  mean={np.mean([r['vis_ent']  for r in recs]):.3f}  "
              f"std={np.std([r['vis_ent']  for r in recs]):.3f}")
        print(f"  n_visited mean={np.mean([r['n_visited'] for r in recs]):.1f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors = {32: "#26A69A", 64: "#E76F51"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = [
        ("imp_ent",  "Improved policy entropy (nats)", "Higher = flatter training target"),
        ("imp_max",  "Improved policy max prob",       "Lower = more uniform target"),
        ("vis_ent",  "Visit count entropy (nats)",     "Higher = more spread visits"),
        ("n_visited","# cells visited",                ""),
    ]
    for ax, (key, title, subtitle) in zip(axes, metrics):
        for sims in budgets:
            vals = [r[key] for r in all_records[sims]]
            p = all_params[sims]
            label = (f"{sims} sims  "
                     f"(k₀={p['k_initial']}, {p['num_phases']} phases, "
                     f"{p['spa0']} sims/cand phase0)")
            ax.hist(vals, bins=40, alpha=0.55, color=colors[sims], label=label,
                    density=True, edgecolor="white", linewidth=0.4)
            ax.axvline(np.mean(vals), color=colors[sims], lw=2, ls="--")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")
        ax.set_xlabel(key, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Gumbel Sequential Halving: sims=32 vs sims=64\n"
                 "9×9 Gomoku, random-ish model, 100 episodes",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.out}")

if __name__ == "__main__":
    main()
