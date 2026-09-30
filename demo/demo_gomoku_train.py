"""
demo_gomoku_train.py — Self-play training: PUCT vs Gumbel Dense on 9×9 Gomoku.

Trains two identical models from scratch using self-play:
  - Model A: PUCT   → policy target = normalized visit counts
  - Model B: Gumbel → policy target = improved policy π' (softmax(logits + σ(Q)))

Both models are evaluated periodically against a 1-ply heuristic opponent
(greedy sliding-window threat scorer). At the end, the two trained models
play each other head-to-head.

Usage:
    uv run python demo/demo_gomoku_train.py
    uv run python demo/demo_gomoku_train.py --games 300 --sims 32 --eval-games 30
"""

import argparse
import random
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

from gumbel_mcts import GumbelDense, PUCT

# ── 9×9 Gomoku ───────────────────────────────────────────────────────────────

BS = 9
NA = BS * BS  # 81

TEMP_MOVES      = 20
BOOTSTRAP_W     = 0.2
TEMPORAL_DECAY  = 0.97
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPS   = 0.15


@njit(cache=False)
def _count_dir(board, r, c, dr, dc, player):
    count = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < BS and 0 <= cc < BS and board[rr, cc] == player:
        count += 1; rr += dr; cc += dc
    return count


@njit(cache=False)
def _check_win(board, r, c, player):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if 1 + _count_dir(board, r, c, dr, dc, player) + \
               _count_dir(board, r, c, -dr, -dc, player) >= 5:
            return True
    return False


@njit(cache=False)
def _fast_step(board, action, player):
    r, c = action // BS, action % BS
    board[r, c] = player
    if _check_win(board, r, c, player):
        return 1.0, player, True, board
    for i in range(BS):
        for j in range(BS):
            if board[i, j] == 0:
                return 0.0, 0, False, board
    return 0.0, 0, True, board


@njit(cache=False)
def _valid_mask(board, player):
    mask = np.zeros(NA, dtype=np.float32)
    for r in range(BS):
        for c in range(BS):
            if board[r, c] == 0:
                mask[r * BS + c] = 1.0
    return mask


class Gomoku9Logic:
    NUM_ACTIONS    = NA
    BOARD_SHAPE    = (BS, BS)
    MAX_MOVES      = NA
    MAX_LEGAL_MOVES = NA
    PLAYER_1       = 1
    PLAYER_2       = 2
    fast_step      = staticmethod(_fast_step)
    get_valid_mask = staticmethod(_valid_mask)


# ── Neural network ────────────────────────────────────────────────────────────

class GomokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc          = nn.Sequential(nn.Linear(64 * BS * BS, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, NA)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 2, BS, BS) — returns raw logits + tanh value
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return self.policy_head(h), torch.tanh(self.value_head(h))


def _encode(boards_raw, player_ids):
    """(B, 81) raw board + (B,) player ids → (B, 2, BS, BS) two-plane spatial tensor."""
    B = boards_raw.shape[0]
    x = torch.zeros(B, 2, BS, BS)
    for i in range(B):
        p     = int(player_ids[i])
        board = boards_raw[i].reshape(BS, BS)
        x[i, 0] = (board == p).float()
        x[i, 1] = (board == (3 - p)).float()
    return x


class GomokuModel:
    def __init__(self, net: GomokuNet, logic: Gomoku9Logic):
        self.net   = net
        self.logic = logic

    def forward_for_mcts(self, batch):
        x = _encode(batch["boards"], batch["current_player"])
        logits, value = self.net(x)
        return {"policy": torch.softmax(logits, dim=-1), "value": value}


class NoisyRootModel:
    """Wraps a GomokuModel and injects Dirichlet noise at the root (PUCT only)."""
    def __init__(self, model: GomokuModel):
        self.model = model
        self.logic = model.logic
        self._root_done = False

    def reset(self): self._root_done = False

    def forward_for_mcts(self, batch):
        out = self.model.forward_for_mcts(batch)
        if not self._root_done:
            self._root_done = True
            pol = out["policy"].clone()
            B   = pol.shape[0]
            noise = torch.from_numpy(
                np.random.dirichlet([DIRICHLET_ALPHA] * NA, size=B).astype(np.float32))
            pol = (1 - DIRICHLET_EPS) * pol + DIRICHLET_EPS * noise
            pol = pol / pol.sum(-1, keepdim=True).clamp(min=1e-8)
            return {"policy": pol, "value": out["value"]}
        return out


class RandomModel:
    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        return {"policy": torch.ones(B, NA) / NA, "value": torch.zeros(B, 1)}


# ── Minimax opponent (alpha-beta) ─────────────────────────────────────────────

@njit(cache=False)
def _eval_board9(board, player):
    """Sliding-window threat scorer for 9×9 (5-in-a-row)."""
    opponent = 3 - player
    score = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(BS):
            for c in range(BS):
                er = r + 4 * dr
                ec = c + 4 * dc
                if er < 0 or er >= BS or ec < 0 or ec >= BS:
                    continue
                pc = 0; oc = 0
                for k in range(5):
                    cell = board[r + k * dr, c + k * dc]
                    if cell == player:     pc += 1
                    elif cell == opponent: oc += 1
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
def _minimax_score(board, current_player, original_player, depth, alpha, beta):
    """Alpha-beta score from original_player's perspective."""
    if depth == 0:
        return _eval_board9(board, original_player)

    mask = _valid_mask(board, current_player)
    maximizing = current_player == original_player
    best = -1e18 if maximizing else 1e18

    for action in range(NA):
        if mask[action] == 0.0:
            continue
        b = board.copy()
        _, winner, done, b = _fast_step(b, action, current_player)
        if done:
            val = 1e9 if winner == original_player else (-1e9 if winner != 0 else 0.0)
        else:
            val = _minimax_score(b, 3 - current_player, original_player, depth - 1, alpha, beta)
        if maximizing:
            if val > best: best = val
            if best > alpha: alpha = best
            if beta <= alpha: break
        else:
            if val < best: best = val
            if best < beta: beta = best
            if beta <= alpha: break

    return best


@njit(cache=False)
def minimax_action(board, player, depth):
    """Return the best move for `player` using alpha-beta minimax."""
    mask  = _valid_mask(board, player)
    best_score  = -1e18
    best_action = -1
    alpha = -1e18

    for action in range(NA):
        if mask[action] == 0.0:
            continue
        b = board.copy()
        _, winner, done, b = _fast_step(b, action, player)
        if done:
            if winner == player:
                return action  # immediate win — take it
            val = 0.0
        else:
            val = _minimax_score(b, 3 - player, player, depth - 1, alpha, 1e18)
        if val > best_score:
            best_score  = val
            best_action = action
        if best_score > alpha:
            alpha = best_score

    return best_action


# ── Self-play data collection ─────────────────────────────────────────────────

def _augment_positions(positions):
    """Apply all 8 dihedral symmetries (4 rotations × 2 reflections) to each position.
    positions: list of (board_flat, player, pol_flat, value)
    """
    augmented = []
    for b_flat, p, pt_flat, v in positions:
        board_2d = b_flat.reshape(BS, BS)
        pol_2d   = pt_flat.reshape(BS, BS)
        for k in range(4):
            rb = np.rot90(board_2d, k)
            rp = np.rot90(pol_2d,   k)
            augmented.append((rb.flatten().copy(), p, rp.flatten().copy(), v))
            fb = np.fliplr(rb)
            fp = np.fliplr(rp)
            augmented.append((fb.flatten().copy(), p, fp.flatten().copy(), v))
    return augmented


def collect_episode(algo: str, sims: int, logic: Gomoku9Logic,
                    model: GomokuModel, noisy_model=None,
                    max_considered_actions: int = 16, entropy_min_k: int = 4):
    """Play one self-play game.

    Returns (augmented_positions, stats) where each position is
    (board_flat, player, train_pol, value_target).
    """
    board     = np.zeros((BS, BS), dtype=np.int8)
    player    = 1
    history   = []   # (board_flat, player, train_pol, root_q)
    opening_move = None
    max_nodes = max(sims * 5 + 200, 1000)
    root_ents  = []   # Shannon entropy of root priors per move (Gumbel only)
    dynamic_ks = []   # dynamic max_k chosen by entropy scheduler per move (Gumbel only)

    model.net.eval()

    for move_num in range(logic.MAX_MOVES):
        if algo == "gumbel":
            tree = GumbelDense(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu",
                               max_considered_actions=max_considered_actions,
                               entropy_min_k=entropy_min_k)
        else:
            tree = PUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([player]))

        if noisy_model is not None:
            noisy_model.reset()
            tree.run_simulation_batch(noisy_model, [0], num_simulations=sims)
        else:
            tree.run_simulation_batch(model, [0], num_simulations=sims)

        if algo == "gumbel" and tree.last_root_entropy is not None:
            root_ents.append(tree.last_root_entropy)
            dynamic_ks.append(tree.last_dynamic_max_k)

        visits, root_q_arr = tree.get_all_root_data(n_active=1)
        root_q = float(root_q_arr[0]) if root_q_arr is not None else 0.0

        # sample_pol: visit counts → action selection
        v = visits[0].astype(np.float32)
        sample_pol = v / v.sum() if v.sum() > 0 else np.ones(NA, dtype=np.float32) / NA
        valid      = logic.get_valid_mask(board, player)
        sample_pol = sample_pol * valid
        sample_pol = (sample_pol / sample_pol.sum()
                      if sample_pol.sum() > 0 else valid / valid.sum())

        # train_pol: improved_policy for Gumbel, visit counts for PUCT
        if algo == "gumbel":
            train_pol = tree.get_improved_policy(n_active=1)[0].astype(np.float32)
            train_pol = train_pol * valid
            train_pol = (train_pol / train_pol.sum()
                         if train_pol.sum() > 0 else valid / valid.sum())
        else:
            train_pol = sample_pol.copy()

        history.append((board.flatten().copy(), player, train_pol, root_q))

        # temperature schedule: sample early, argmax later
        if move_num < TEMP_MOVES:
            action = int(np.random.choice(NA, p=sample_pol))
        else:
            action = int(np.argmax(sample_pol))

        if opening_move is None:
            opening_move = action

        _, winner, done, board = logic.fast_step(board, action, player)
        if done:
            break
        player = 3 - player

    L = len(history)
    positions     = []
    total_entropy = 0.0
    for t, (b_flat, p, pt, q) in enumerate(history):
        outcome        = 1.0 if winner == p else (-1.0 if winner == (3 - p) else 0.0)
        moves_from_end = L - 1 - t
        eff_boot       = BOOTSTRAP_W * (TEMPORAL_DECAY ** moves_from_end)
        value_target   = (1 - eff_boot) * outcome + eff_boot * q
        positions.append((b_flat, p, pt, value_target))
        total_entropy += -float(np.sum(pt * np.log(pt + 1e-8)))

    stats = {
        "game_length":        L,
        "opening_move":       opening_move if opening_move is not None else 0,
        "pol_entropy":        total_entropy / max(L, 1),
        "mean_root_entropy":  float(np.mean(root_ents))  if root_ents  else None,
        "mean_dynamic_max_k": float(np.mean(dynamic_ks)) if dynamic_ks else None,
    }
    return _augment_positions(positions), stats


# ── Statistics helpers ────────────────────────────────────────────────────────

def wilson_ci(wins, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 100.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, (centre - spread) * 100), min(100.0, (centre + spread) * 100)


# ── Training ──────────────────────────────────────────────────────────────────

def train_step(net: GomokuNet, optimizer, buffer, batch_size: int = 64):
    batch = random.sample(buffer, min(batch_size, len(buffer)))
    boards, player_ids, pol_targets, val_targets = zip(*batch)

    boards_t = torch.tensor(np.stack(boards), dtype=torch.float32)
    pids_t   = torch.tensor(player_ids,       dtype=torch.long)
    x        = _encode(boards_t, pids_t)
    tp       = torch.tensor(np.stack(pol_targets), dtype=torch.float32)
    tv       = torch.tensor(val_targets,           dtype=torch.float32).unsqueeze(1)

    # valid mask: empty cells (board value == 0), shape (B, 81)
    valid = (boards_t == 0).float()

    net.train()
    logits, pred_v = net(x)
    # mask illegal moves with -inf, then fused log_softmax for numerical stability
    logits    = logits.masked_fill(valid == 0, float("-inf"))
    log_pred_p = F.log_softmax(logits, dim=-1)
    # zero out illegal positions: tp is 0 there, but 0 * -inf = nan in IEEE 754
    log_pred_p = log_pred_p.masked_fill(valid == 0, 0.0)
    loss = -(tp * log_pred_p).sum(-1).mean() + F.mse_loss(pred_v, tv)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_vs_minimax(algo: str, net: GomokuNet, logic: Gomoku9Logic,
                        n_games: int, sims: int, minimax_depth: int,
                        minimax_eps: float = 0.0,
                        max_considered_actions: int = 16,
                        entropy_min_k: int = 4) -> float:
    """Win rate of trained model vs an epsilon-greedy alpha-beta minimax opponent.

    With probability minimax_eps the opponent plays a random legal move instead
    of its minimax best move — making it beatable during early training.
    """
    model = GomokuModel(net, logic)
    net.eval()
    wins = 0

    for g in range(n_games):
        board          = np.zeros((BS, BS), dtype=np.int8)
        player         = 1
        trained_player = 1 if g % 2 == 0 else 2
        max_nodes      = max(sims * 5 + 200, 1000)

        for _ in range(logic.MAX_MOVES):
            if player == trained_player:
                if algo == "gumbel":
                    tree = GumbelDense(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu",
                                       max_considered_actions=max_considered_actions,
                                       entropy_min_k=entropy_min_k)
                else:
                    tree = PUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                with torch.no_grad():
                    tree.run_simulation_batch(model, [0], num_simulations=sims)
                if algo == "gumbel":
                    pol = tree.get_improved_policy(n_active=1)[0].astype(np.float32)
                else:
                    visits, _ = tree.get_all_root_data(n_active=1)
                    pol = visits[0].astype(np.float32)
                action = int(np.argmax(pol))
            else:
                if minimax_eps > 0.0 and np.random.random() < minimax_eps:
                    legal  = np.where(logic.get_valid_mask(board, player) > 0)[0]
                    action = int(np.random.choice(legal))
                else:
                    action = int(minimax_action(board, player, minimax_depth))

            _, winner, done, board = logic.fast_step(board, action, player)
            if done:
                if winner == trained_player:
                    wins += 1
                break
            player = 3 - player

    return wins / n_games * 100.0, wins


def head_to_head(gumbel_net: GomokuNet, puct_net: GomokuNet,
                 logic: Gomoku9Logic, n_games: int, sims: int,
                 max_considered_actions: int = 16,
                 entropy_min_k: int = 4) -> float:
    """Gumbel-trained vs PUCT-trained. Returns Gumbel win rate."""
    g_model = GomokuModel(gumbel_net, logic)
    p_model = GomokuModel(puct_net,   logic)
    gumbel_net.eval(); puct_net.eval()
    gumbel_wins = 0
    max_nodes   = max(sims * 5 + 200, 1000)

    for g in range(n_games):
        board         = np.zeros((BS, BS), dtype=np.int8)
        player        = 1
        gumbel_player = 1 if g % 2 == 0 else 2

        for _ in range(logic.MAX_MOVES):
            if player == gumbel_player:
                tree = GumbelDense(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu",
                                   max_considered_actions=max_considered_actions,
                                   entropy_min_k=entropy_min_k)
                tree.initialize_roots([0], board[None], np.array([player]))
                with torch.no_grad():
                    tree.run_simulation_batch(g_model, [0], num_simulations=sims)
                action = int(np.argmax(
                    tree.get_improved_policy(n_active=1)[0]))
            else:
                tree = PUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                with torch.no_grad():
                    tree.run_simulation_batch(p_model, [0], num_simulations=sims)
                visits, _ = tree.get_all_root_data(n_active=1)
                action = int(np.argmax(visits[0]))

            _, winner, done, board = logic.fast_step(board, action, player)
            if done:
                if winner == gumbel_player:
                    gumbel_wins += 1
                break
            player = 3 - player

    return gumbel_wins / n_games * 100.0, gumbel_wins


# ── Training loop ─────────────────────────────────────────────────────────────

def train_system(algo: str, args, logic: Gomoku9Logic, sims: int,
                 entropy_min_k: int = 4):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    net         = GomokuNet()
    optimizer   = torch.optim.Adam(net.parameters(), lr=1e-3)
    model       = GomokuModel(net, logic)
    noisy_model = NoisyRootModel(model) if algo == "puct" else None
    buffer      = deque(maxlen=args.replay_size)
    curve       = []

    div_lengths, div_openings, div_pol_ents = [], [], []
    div_root_ents, div_dynamic_ks = [], []

    t_start      = time.time()
    progress_every = max(10, args.games // 20)   # ~20 lightweight progress ticks

    for ep in range(1, args.games + 1):
        data, stats = collect_episode(algo, sims, logic, model, noisy_model,
                                      args.max_considered_actions, entropy_min_k)
        buffer.extend(data)
        div_lengths.append(stats["game_length"])
        div_openings.append(stats["opening_move"])
        div_pol_ents.append(stats["pol_entropy"])
        if stats["mean_root_entropy"] is not None:
            div_root_ents.append(stats["mean_root_entropy"])
            div_dynamic_ks.append(stats["mean_dynamic_max_k"])

        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps):
                train_step(net, optimizer, list(buffer), args.batch_size)

        if ep % progress_every == 0 and ep % args.eval_every != 0:
            elapsed = time.time() - t_start
            eta     = elapsed / ep * (args.games - ep)
            print(f"    ep={ep}/{args.games}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", flush=True)

        if ep % args.eval_every == 0:
            net.eval()
            with torch.no_grad():
                wr, wr_wins = evaluate_vs_minimax(algo, net, logic, args.eval_games, sims,
                                                   args.minimax_depth, args.minimax_eps,
                                                   args.max_considered_actions, entropy_min_k)
            net.train()
            curve.append((ep, wr))
            ci_lo, ci_hi = wilson_ci(wr_wins, args.eval_games)

            col_counts = np.bincount(div_openings, minlength=NA)
            top5 = np.argsort(col_counts)[::-1][:5]
            top5_str = " ".join(f"({i//BS},{i%BS}):{col_counts[i]}" for i in top5)
            # Opening diversity: entropy over first-move distribution + unique count
            n_uniq_opens = int((col_counts > 0).sum())
            open_probs   = col_counts / col_counts.sum()
            open_ent     = float(-np.sum(open_probs * np.log(open_probs + 1e-12)))
            elapsed = time.time() - t_start
            eta     = elapsed / ep * (args.games - ep)
            if div_root_ents:
                # Gumbel: show raw-prior entropy + scheduler's chosen k + improved-policy entropy
                print(f"  ep={ep:4d}/{args.games}  win={wr:.1f}% [{ci_lo:.0f}%-{ci_hi:.0f}%]  "
                      f"len={np.mean(div_lengths):.1f}  "
                      f"pol_ent={np.mean(div_pol_ents):.2f}  "
                      f"root_ent={np.mean(div_root_ents):.2f}  "
                      f"avg_k={np.mean(div_dynamic_ks):.1f}  "
                      f"open_ent={open_ent:.2f}  uniq_opens={n_uniq_opens}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                      f"top5_opens=[{top5_str}]")
            else:
                # PUCT: no entropy scheduler
                print(f"  ep={ep:4d}/{args.games}  win={wr:.1f}% [{ci_lo:.0f}%-{ci_hi:.0f}%]  "
                      f"len={np.mean(div_lengths):.1f}  "
                      f"ent={np.mean(div_pol_ents):.2f}  "
                      f"open_ent={open_ent:.2f}  uniq_opens={n_uniq_opens}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                      f"top5_opens=[{top5_str}]")
            div_lengths.clear(); div_openings.clear(); div_pol_ents.clear()
            div_root_ents.clear(); div_dynamic_ks.clear()

    return net, curve


# ── Plot ──────────────────────────────────────────────────────────────────────

# Color per sim budget (light → dark as budget increases)
_BUDGET_COLORS = ["#F4A261", "#E76F51", "#264653"]   # up to 3 budgets; extend if needed


def plot_results(results: dict, sim_budgets: list, minimax_eps: float, out_path: str):
    """
    results: {(algo, sims): [(ep, win_rate), ...]}
    Produces two subplots:
      Left  — learning curves (Gumbel solid, PUCT dashed, color = sim budget)
      Right — final win rate grouped bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Learning curves ───────────────────────────────────────────────────────
    ax = axes[0]
    for i, sims in enumerate(sim_budgets):
        color = _BUDGET_COLORS[i % len(_BUDGET_COLORS)]
        for algo, ls, marker in [("gumbel", "-", "s"), ("puct", "--", "o")]:
            curve = results.get((algo, sims), [])
            if not curve:
                continue
            ep = [c[0] for c in curve]
            wr = [c[1] for c in curve]
            label = f"{'Gumbel' if algo == 'gumbel' else 'PUCT'} · {sims} sims"
            ax.plot(ep, wr, ls, marker=marker, color=color, label=label,
                    lw=2, ms=6, mfc="white", mew=1.8, alpha=0.9)

    ax.axhline(50, color="#BDBDBD", ls=":", lw=1)
    ax.set_xlabel("Self-play episodes", fontsize=11)
    ax.set_ylabel(f"Win rate vs minimax(d=2, ε={minimax_eps}) (%)", fontsize=10)
    ax.set_title("Learning curves — 9×9 Gomoku self-play\n"
                 "Solid = Gumbel Dense, Dashed = PUCT",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, ncol=2)

    # ── Final win rate bar chart ──────────────────────────────────────────────
    ax2 = axes[1]
    n  = len(sim_budgets)
    w  = 0.35
    xs = np.arange(n)

    gumbel_finals = [results.get(("gumbel", s), [(0, 0)])[-1][1] for s in sim_budgets]
    puct_finals   = [results.get(("puct",   s), [(0, 0)])[-1][1] for s in sim_budgets]

    bars_g = ax2.bar(xs - w / 2, gumbel_finals, w, label="Gumbel Dense",
                     color="#26A69A", alpha=0.85, edgecolor="white")
    bars_p = ax2.bar(xs + w / 2, puct_finals,   w, label="PUCT",
                     color="#5C6BC0", alpha=0.85, edgecolor="white")
    for bar, val in list(zip(bars_g, gumbel_finals)) + list(zip(bars_p, puct_finals)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    ax2.axhline(50, color="#BDBDBD", ls=":", lw=1)
    ax2.set_xticks(xs); ax2.set_xticklabels([f"{s} sims" for s in sim_budgets])
    ax2.set_ylabel("Final win rate vs minimax (%)", fontsize=11)
    ax2.set_title("Final win rate by sim budget", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115); ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",        type=int,   default=200,
                    help="self-play episodes per (algo, budget) run (default 200)")
    ap.add_argument("--sim-budgets",  type=str,   default="8,16,32",
                    help="comma-separated sim budgets to sweep (default 8,16,32)")
    ap.add_argument("--eval-games",   type=int,   default=100,
                    help="evaluation games vs minimax per checkpoint (default 100)")
    ap.add_argument("--eval-every",   type=int,   default=50,
                    help="evaluate every N episodes (default 50)")
    ap.add_argument("--train-steps",  type=int,   default=2,
                    help="gradient steps per episode (default 2)")
    ap.add_argument("--batch-size",   type=int,   default=64)
    ap.add_argument("--replay-size",  type=int,   default=2000)
    ap.add_argument("--h2h-games",    type=int,   default=200,
                    help="head-to-head games per budget at the end (default 200)")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--minimax-depth", type=int,  default=2,
                    help="alpha-beta depth for the eval opponent (default 2)")
    ap.add_argument("--minimax-eps",  type=float, default=0.3,
                    help="epsilon for epsilon-greedy minimax opponent (default 0.3)")
    ap.add_argument("--max-considered-actions", type=int, default=16,
                    help="max candidates for Gumbel Sequential Halving (default 16)")
    ap.add_argument("--entropy-min-k", type=int, default=4,
                    help="min candidates used when model is fully random (default 4); "
                         "entropy scheduler interpolates between this and --max-considered-actions")
    ap.add_argument("--out",          type=str,   default="demo/gomoku_train_curves.png")
    args = ap.parse_args()

    sim_budgets = [int(x) for x in args.sim_budgets.split(",")]
    logic       = Gomoku9Logic()

    # warm up numba JIT
    dummy = np.zeros((BS, BS), dtype=np.int8)
    _fast_step(dummy.copy(), 0, 1); _valid_mask(dummy, 1)
    print("Compiling minimax kernel (first call only)...")
    minimax_action(dummy, 1, 1)
    print("Numba kernels ready.\n")
    print(f"9×9 Gomoku — Gumbel vs PUCT, sim budgets={sim_budgets}")
    print(f"  episodes={args.games}  eval_every={args.eval_every}")
    print(f"  minimax depth={args.minimax_depth}  ε={args.minimax_eps}\n")

    results = {}   # {(algo, sims): curve}
    nets    = {}   # {(algo, sims): net}  for head-to-head

    for sims in sim_budgets:
        for algo in ["gumbel", "puct"]:
            print(f"\n=== {algo.upper()} · {sims} sims ===")
            net, curve       = train_system(algo, args, logic, sims, args.entropy_min_k)
            results[(algo, sims)] = curve
            nets[(algo, sims)]    = net

    # Head-to-head per budget
    print(f"\n=== Head-to-head (Gumbel-trained vs PUCT-trained, n={args.h2h_games} games each) ===")
    for sims in sim_budgets:
        gwr, g_wins = head_to_head(nets[("gumbel", sims)], nets[("puct", sims)],
                                   logic, args.h2h_games, sims,
                                   args.max_considered_actions, args.entropy_min_k)
        p_wins = args.h2h_games - g_wins
        g_lo, g_hi = wilson_ci(g_wins, args.h2h_games)
        p_lo, p_hi = wilson_ci(p_wins, args.h2h_games)
        print(f"  {sims:3d} sims:  "
              f"Gumbel {gwr:.1f}% [{g_lo:.0f}%-{g_hi:.0f}%]  "
              f"PUCT {100-gwr:.1f}% [{p_lo:.0f}%-{p_hi:.0f}%]")

    plot_results(results, sim_budgets, args.minimax_eps, args.out)


if __name__ == "__main__":
    main()
