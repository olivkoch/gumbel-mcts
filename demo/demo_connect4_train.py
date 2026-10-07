"""
Self-play training comparison: PUCT vs Gumbel Dense on Connect 4.
Trains both algorithms with varying sim budgets, evaluates against
a depth-2 epsilon-minimax opponent, then plots learning curves.
"""

import argparse
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

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS = 6, 7
NA = COLS   # 7 actions (one per column)

# Self-play hyperparameters (from game_gen analysis)
TEMP_MOVES     = 15    # sample from visits for first N moves, then argmax
BOOTSTRAP_W    = 0.2   # weight on MCTS Q-value in value target
TEMPORAL_DECAY = 0.97  # decay by moves-from-end (early positions trust outcome more)
DIRICHLET_ALPHA = 0.15 # Dirichlet concentration (for PUCT only)
DIRICHLET_EPS   = 0.15 # noise mixing coefficient (for PUCT only) — lower than alphazero's
                       # 0.35 because 32 sims produces weaker visit signal than their 50

def wilson_ci(wins, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 100.0
    p = wins / n
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, (centre - spread) * 100), min(100.0, (centre + spread) * 100)

# ── Game kernels ───────────────────────────────────────────────────────────────

@njit(cache=False)
def _check_win(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if board[r,c]==player and board[r,c+1]==player and board[r,c+2]==player and board[r,c+3]==player:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if board[r,c]==player and board[r+1,c]==player and board[r+2,c]==player and board[r+3,c]==player:
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if board[r,c]==player and board[r-1,c+1]==player and board[r-2,c+2]==player and board[r-3,c+3]==player:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if board[r,c]==player and board[r+1,c+1]==player and board[r+2,c+2]==player and board[r+3,c+3]==player:
                return True
    return False


@njit(cache=False)
def _fast_step(board, action, player):
    board = board.copy()
    row = -1
    for r in range(ROWS - 1, -1, -1):
        if board[r, action] == 0:
            row = r
            break
    if row == -1:
        return 0.0, 0, False, board
    board[row, action] = player
    if _check_win(board, player):
        return 1.0, player, True, board
    for c in range(COLS):
        if board[0, c] == 0:
            return 0.0, 0, False, board
    return 0.0, 0, True, board   # board full → draw


@njit(cache=False)
def _valid_mask(board, player):
    mask = np.zeros(COLS, dtype=np.float32)
    for c in range(COLS):
        if board[0, c] == 0:
            mask[c] = 1.0
    return mask

# ── Depth-2 alpha-beta minimax ─────────────────────────────────────────────────

@njit(cache=False)
def _eval_board(board, player):
    """Count 3-in-a-row threats as a leaf heuristic."""
    opp   = 3 - player
    score = 0
    # Iterate all directions: horizontal, vertical, diag \, diag /
    for r in range(ROWS):
        for c in range(COLS - 3):
            p = o = 0
            for k in range(4):
                cell = board[r, c + k]
                if cell == player: p += 1
                elif cell == opp:  o += 1
            if o == 0 and p == 3: score += 5
            elif o == 0 and p == 2: score += 2
            elif p == 0 and o == 3: score -= 4
    for r in range(ROWS - 3):
        for c in range(COLS):
            p = o = 0
            for k in range(4):
                cell = board[r + k, c]
                if cell == player: p += 1
                elif cell == opp:  o += 1
            if o == 0 and p == 3: score += 5
            elif o == 0 and p == 2: score += 2
            elif p == 0 and o == 3: score -= 4
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p = o = 0
            for k in range(4):
                cell = board[r + k, c + k]
                if cell == player: p += 1
                elif cell == opp:  o += 1
            if o == 0 and p == 3: score += 5
            elif o == 0 and p == 2: score += 2
            elif p == 0 and o == 3: score -= 4
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            p = o = 0
            for k in range(4):
                cell = board[r - k, c + k]
                if cell == player: p += 1
                elif cell == opp:  o += 1
            if o == 0 and p == 3: score += 5
            elif o == 0 and p == 2: score += 2
            elif p == 0 and o == 3: score -= 4
    return score


@njit(cache=False)
def _minimax_score(board, current_player, original_player, depth, alpha, beta):
    if _check_win(board, original_player):     return 10000
    if _check_win(board, 3 - original_player): return -10000
    full = True
    for c in range(COLS):
        if board[0, c] == 0:
            full = False
            break
    if full or depth == 0:
        return _eval_board(board, original_player)

    opp = 3 - current_player
    if current_player == original_player:   # maximizing
        best = -99999
        for c in range(COLS):
            if board[0, c] == 0:
                row = -1
                for r in range(ROWS - 1, -1, -1):
                    if board[r, c] == 0: row = r; break
                board[row, c] = current_player
                val = _minimax_score(board, opp, original_player, depth - 1, alpha, beta)
                board[row, c] = 0
                if val > best: best = val
                if best > alpha: alpha = best
                if alpha >= beta: break
        return best
    else:                                   # minimizing
        best = 99999
        for c in range(COLS):
            if board[0, c] == 0:
                row = -1
                for r in range(ROWS - 1, -1, -1):
                    if board[r, c] == 0: row = r; break
                board[row, c] = current_player
                val = _minimax_score(board, opp, original_player, depth - 1, alpha, beta)
                board[row, c] = 0
                if val < best: best = val
                if best < beta: beta = best
                if alpha >= beta: break
        return best


@njit(cache=False)
def minimax_action(board, player, depth):
    best_score  = -99999
    best_action = 0
    n_best      = 0
    opp = 3 - player
    for c in range(COLS):
        if board[0, c] == 0:
            row = -1
            for r in range(ROWS - 1, -1, -1):
                if board[r, c] == 0: row = r; break
            board[row, c] = player
            val = _minimax_score(board, opp, player, depth - 1, -99999, 99999)
            board[row, c] = 0
            if val > best_score:
                best_score  = val
                best_action = c
                n_best      = 1
            elif val == best_score:
                n_best += 1
                if np.random.randint(0, n_best) == 0:  # reservoir sampling over ties
                    best_action = c
    return best_action

# ── GameLogic ──────────────────────────────────────────────────────────────────

class Connect4Logic:
    NUM_ACTIONS = NA
    BOARD_SHAPE  = (ROWS, COLS)
    MAX_MOVES    = ROWS * COLS
    PLAYER_1     = 1
    PLAYER_2     = 2

    fast_step      = staticmethod(_fast_step)
    get_valid_mask = staticmethod(_valid_mask)

# ── Model ──────────────────────────────────────────────────────────────────────

class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ROWS * COLS, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, NA)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        h = self.conv(x)
        return F.softmax(self.policy_head(h), dim=-1), torch.tanh(self.value_head(h))


def _encode(boards_raw, player_ids):
    B = len(boards_raw)
    x = torch.zeros(B, 2, ROWS, COLS)
    for i in range(B):
        p    = int(player_ids[i])
        board = boards_raw[i].reshape(ROWS, COLS)
        x[i, 0] = torch.from_numpy((board == p).astype(np.float32))
        x[i, 1] = torch.from_numpy((board == (3 - p)).astype(np.float32))
    return x


class Connect4Model:
    def __init__(self, net: Connect4Net, device="cpu"):
        self.net    = net
        self.device = device
        self.logic  = Connect4Logic()

    def forward_for_mcts(self, batch):
        boards_raw  = batch["boards"].numpy()          # (B, ROWS*COLS) flattened
        players_raw = batch["current_player"].numpy()
        x = _encode(boards_raw, players_raw).to(self.device)
        with torch.no_grad():
            policy, value = self.net(x)
        # mask illegal moves (full columns) before returning to the MCTS kernel
        B = len(boards_raw)
        mask = torch.zeros(B, NA)
        for i in range(B):
            top_row = boards_raw[i].reshape(ROWS, COLS)[0]
            mask[i] = torch.tensor(top_row == 0, dtype=torch.float32)
        mask   = mask.to(self.device)
        policy = policy * mask
        policy = policy / policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return {"policy": policy, "value": value.squeeze(-1)}

# ── Noisy-root model wrapper (PUCT only) ───────────────────────────────────────

class NoisyRootModel:
    """Injects Dirichlet noise into root priors on the first forward_for_mcts call
    per tree (root expansion). PUCT needs this for exploration diversity; Gumbel
    already explores via its built-in Gumbel sampling at the root."""
    def __init__(self, model: Connect4Model):
        self.model       = model
        self.logic       = model.logic
        self._root_done  = False

    def reset(self):
        self._root_done = False

    def forward_for_mcts(self, batch):
        out = self.model.forward_for_mcts(batch)
        if not self._root_done:
            self._root_done = True
            pol   = out["policy"].clone()
            B     = pol.shape[0]
            noise = torch.from_numpy(
                np.random.dirichlet([DIRICHLET_ALPHA] * NA, size=B).astype(np.float32)
            )
            pol = (1 - DIRICHLET_EPS) * pol + DIRICHLET_EPS * noise
            pol = pol / pol.sum(-1, keepdim=True).clamp(min=1e-8)
            return {"policy": pol, "value": out["value"]}
        return out

# ── Self-play episode ──────────────────────────────────────────────────────────

def _mirror_positions(positions):
    """Horizontal flip augmentation — Connect 4 is left-right symmetric.
    Doubles training data and eliminates column-side bias."""
    mirrored = []
    for b, p, pt, v in positions:
        mirrored.append((
            b[:, ::-1].copy(),   # flip board columns
            p,                   # player unchanged
            pt[::-1].copy(),     # flip action distribution (col 0↔6, 1↔5, 2↔4)
            v,                   # value unchanged
        ))
    return mirrored


def collect_episode(algo: str, sims: int, logic: Connect4Logic,
                    model: Connect4Model, noisy_model=None, args=None):
    """
    Returns (positions, stats) where stats = {game_length, opening_move, pol_entropy}.
    noisy_model: NoisyRootModel used for PUCT; None for Gumbel.
    """
    board   = np.zeros((ROWS, COLS), dtype=np.int8)
    player  = 1
    history = []   # (board, player, train_pol, root_q)
    opening_move = None

    while True:
        if algo == "gumbel":
            tree = GumbelDense(
                n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu",
                max_considered_actions=args.max_considered_actions,
                entropy_min_k=args.entropy_min_k,
            )
        else:
            tree = PUCT(n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([player]))

        # PUCT gets Dirichlet noise at the root; Gumbel uses its own sampling
        if noisy_model is not None:
            noisy_model.reset()
            tree.run_simulation_batch(noisy_model, [0], num_simulations=sims)
        else:
            tree.run_simulation_batch(model, [0], num_simulations=sims)

        # Visit counts → action sampling; root Q → value target mixing
        visits, root_q_arr = tree.get_all_root_data(n_active=1)
        root_q = float(root_q_arr[0]) if root_q_arr is not None else 0.0

        v = visits[0].astype(np.float32)
        sample_pol = v / v.sum() if v.sum() > 0 else np.ones(NA, dtype=np.float32) / NA

        valid = logic.get_valid_mask(board, player)
        sample_pol = sample_pol * valid
        sample_pol = sample_pol / sample_pol.sum() if sample_pol.sum() > 0 else valid / valid.sum()

        # Training target: improved_policy for Gumbel, visit counts for PUCT
        if algo == "gumbel":
            train_pol = tree.get_improved_policy(n_active=1)[0].astype(np.float32)
            train_pol = train_pol * valid
            train_pol = train_pol / train_pol.sum() if train_pol.sum() > 0 else valid / valid.sum()
        else:
            train_pol = sample_pol.copy()

        history.append((board.copy(), player, train_pol.copy(), root_q))

        # Temperature schedule: stochastic for first TEMP_MOVES, greedy after
        move_number = len(history)
        if move_number <= TEMP_MOVES:
            action = int(np.random.choice(NA, p=sample_pol))
        else:
            action = int(np.argmax(sample_pol))

        if opening_move is None:
            opening_move = action

        _, winner, done, board = logic.fast_step(board, action, player)
        if done:
            break
        player = 3 - player

    # Value target: mix game outcome with MCTS Q (reduces variance, speeds learning)
    L = len(history)
    positions = []
    total_entropy = 0.0
    for t, (b, p, pt, q) in enumerate(history):
        outcome        = 1.0 if winner == p else (-1.0 if winner == (3 - p) else 0.0)
        moves_from_end = L - 1 - t
        eff_boot       = BOOTSTRAP_W * (TEMPORAL_DECAY ** moves_from_end)
        value_target   = (1 - eff_boot) * outcome + eff_boot * q
        positions.append((b, p, pt, value_target))
        # Policy entropy for diversity monitoring
        ent = -float(np.sum(pt * np.log(pt + 1e-8)))
        total_entropy += ent

    stats = {
        "game_length":  L,
        "opening_move": opening_move,
        "pol_entropy":  total_entropy / max(L, 1),
    }
    return positions + _mirror_positions(positions), stats

# ── Training ───────────────────────────────────────────────────────────────────

def train_step(net: Connect4Net, optimizer, buffer, batch_size=64):
    if len(buffer) < batch_size:
        return
    idxs       = np.random.choice(len(buffer), batch_size, replace=False)
    batch      = [buffer[i] for i in idxs]
    raw_boards = np.stack([b for b,_,_,_ in batch])
    players    = np.array([p for _,p,_,_ in batch])

    x     = _encode(raw_boards, players)
    pol_t = torch.tensor(np.stack([pt for _,_,pt,_ in batch]), dtype=torch.float32)
    val_t = torch.tensor([v for _,_,_,v in batch], dtype=torch.float32)

    # valid mask: top row == 0 means column is not full
    masks = torch.zeros(batch_size, NA)
    for i in range(batch_size):
        masks[i] = torch.tensor(
            raw_boards[i].reshape(ROWS, COLS)[0] == 0, dtype=torch.float32
        )

    policy, value = net(x)
    masked_policy = policy * masks
    masked_policy = masked_policy / masked_policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    loss = -(pol_t * torch.log(masked_policy + 1e-8)).sum(-1).mean() \
           + F.mse_loss(value.squeeze(-1), val_t)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_vs_minimax(algo: str, net: Connect4Net, logic: Connect4Logic,
                        n_games: int, sims: int, minimax_depth: int,
                        minimax_eps: float = 0.3, args=None):
    model = Connect4Model(net)
    wins  = 0
    for g in range(n_games):
        algo_player = 1 if g % 2 == 0 else 2
        board  = np.zeros((ROWS, COLS), dtype=np.int8)
        player = 1
        done   = False
        while not done:
            if player == algo_player:
                if algo == "gumbel":
                    tree = GumbelDense(
                        n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu",
                        max_considered_actions=args.max_considered_actions,
                        entropy_min_k=args.entropy_min_k,
                    )
                else:
                    tree = PUCT(n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                tree.run_simulation_batch(model, [0], num_simulations=sims)
                if algo == "gumbel":
                    pol = tree.get_improved_policy(n_active=1)[0].astype(np.float32)
                else:
                    visits, _ = tree.get_all_root_data(n_active=1)
                    pol = visits[0].astype(np.float32)
                valid  = logic.get_valid_mask(board, player)
                pol    = pol * valid
                action = int(np.argmax(pol)) if pol.sum() > 0 else int(np.argmax(valid))
            else:
                if minimax_eps > 0.0 and np.random.random() < minimax_eps:
                    legal  = np.where(logic.get_valid_mask(board, player) > 0)[0]
                    action = int(np.random.choice(legal))
                else:
                    action = int(minimax_action(board.copy(), player, minimax_depth))
            _, winner, done, board = logic.fast_step(board, action, player)
            if done:
                if winner == algo_player:
                    wins += 1
                break
            player = 3 - player
    return 100.0 * wins / n_games, wins


def head_to_head(gumbel_net: Connect4Net, puct_net: Connect4Net,
                 logic: Connect4Logic, n_games: int, sims: int, args=None):
    gm = Connect4Model(gumbel_net)
    pm = Connect4Model(puct_net)
    wins = 0
    for g in range(n_games):
        gumbel_player = 1 if g % 2 == 0 else 2
        board  = np.zeros((ROWS, COLS), dtype=np.int8)
        player = 1
        done   = False
        while not done:
            if player == gumbel_player:
                tree = GumbelDense(
                    n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu",
                    max_considered_actions=args.max_considered_actions,
                    entropy_min_k=args.entropy_min_k,
                )
                tree.initialize_roots([0], board[None], np.array([player]))
                tree.run_simulation_batch(gm, [0], num_simulations=sims)
            else:
                tree = PUCT(n_games=1, max_nodes=args.max_nodes, logic=logic, device="cpu")
                tree.initialize_roots([0], board[None], np.array([player]))
                tree.run_simulation_batch(pm, [0], num_simulations=sims)
            if player == gumbel_player:
                pol = tree.get_improved_policy(n_active=1)[0].astype(np.float32)
            else:
                visits, _ = tree.get_all_root_data(n_active=1)
                pol = visits[0].astype(np.float32)
            valid  = logic.get_valid_mask(board, player)
            pol    = pol * valid
            action = int(np.argmax(pol)) if pol.sum() > 0 else int(np.argmax(valid))
            _, winner, done, board = logic.fast_step(board, action, player)
            if done:
                if winner == gumbel_player:
                    wins += 1
                break
            player = 3 - player
    return 100.0 * wins / n_games, wins

# ── Training loop ──────────────────────────────────────────────────────────────

def train_system(algo: str, args, logic: Connect4Logic, sims: int):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net         = Connect4Net()
    model       = Connect4Model(net)
    noisy_model = NoisyRootModel(model) if algo == "puct" else None
    optimizer   = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer      = deque(maxlen=args.replay_size)
    curve       = []

    div_lengths, div_openings, div_entropies = [], [], []
    t_start        = time.time()
    progress_every = max(10, args.games // 20)

    for ep in range(1, args.games + 1):
        positions, stats = collect_episode(algo, sims, logic, model, noisy_model, args)
        buffer.extend(positions)
        div_lengths.append(stats["game_length"])
        div_openings.append(stats["opening_move"])
        div_entropies.append(stats["pol_entropy"])

        for _ in range(args.train_steps):
            train_step(net, optimizer, buffer, args.batch_size)

        if ep % args.eval_every == 0:
            net.eval()
            with torch.no_grad():
                wr, wr_wins = evaluate_vs_minimax(algo, net, logic, args.eval_games, sims,
                                                   args.minimax_depth, args.minimax_eps, args)
            net.train()
            curve.append((ep, wr))
            ci_lo, ci_hi = wilson_ci(wr_wins, args.eval_games)

            col_counts   = np.bincount(div_openings, minlength=NA)
            col_dist     = " ".join(f"c{c}:{col_counts[c]}" for c in range(NA))
            n_uniq_opens = int((col_counts > 0).sum())
            open_probs   = col_counts / col_counts.sum()
            open_ent     = float(-np.sum(open_probs * np.log(open_probs + 1e-12)))
            elapsed      = time.time() - t_start
            eta          = elapsed / ep * (args.games - ep)
            print(f"  ep={ep:4d}/{args.games}  win={wr:.1f}% [{ci_lo:.0f}%-{ci_hi:.0f}%]  "
                  f"len={np.mean(div_lengths):.1f}  "
                  f"ent={np.mean(div_entropies):.2f}  "
                  f"open_ent={open_ent:.2f}  uniq_opens={n_uniq_opens}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                  f"openings=[{col_dist}]")
            div_lengths.clear(); div_openings.clear(); div_entropies.clear()
        elif ep % progress_every == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / ep * (args.games - ep)
            print(f"    ep={ep}/{args.games}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                  flush=True)

    return net, curve

# ── Plot ───────────────────────────────────────────────────────────────────────

_COLORS = ["#F4A261", "#E76F51", "#E9C46A", "#264653", "#2A9D8F", "#A8DADC"]


def plot_results(results: dict, sim_budgets: list, minimax_eps: float, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for i, sims in enumerate(sim_budgets):
        color = _COLORS[i % len(_COLORS)]
        for algo, ls, marker in [("gumbel", "-", "s"), ("puct", "--", "o")]:
            curve = results.get((algo, sims), [])
            if not curve:
                continue
            ep = [c[0] for c in curve]
            wr = [c[1] for c in curve]
            ax.plot(ep, wr, ls, marker=marker, color=color,
                    label=f"{'Gumbel' if algo=='gumbel' else 'PUCT'} · {sims} sims",
                    lw=2, ms=6, mfc="white", mew=1.8, alpha=0.9)
    ax.axhline(50, color="#BDBDBD", ls=":", lw=1)
    ax.set_xlabel("Self-play episodes", fontsize=11)
    ax.set_ylabel(f"Win rate vs minimax(d=2, ε={minimax_eps}) (%)", fontsize=10)
    ax.set_title("Learning curves — Connect 4 self-play\n"
                 "Solid = Gumbel Dense, Dashed = PUCT", fontsize=11, fontweight="bold")
    ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, ncol=2)

    ax2   = axes[1]
    n, w  = len(sim_budgets), 0.35
    xs    = np.arange(n)
    gf    = [results.get(("gumbel", s), [(0, 0)])[-1][1] for s in sim_budgets]
    pf    = [results.get(("puct",   s), [(0, 0)])[-1][1] for s in sim_budgets]
    bars_g = ax2.bar(xs - w/2, gf, w, label="Gumbel Dense", color="#26A69A", alpha=0.85, edgecolor="white")
    bars_p = ax2.bar(xs + w/2, pf, w, label="PUCT",         color="#5C6BC0", alpha=0.85, edgecolor="white")
    for bar, val in list(zip(bars_g, gf)) + list(zip(bars_p, pf)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
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

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",                type=int,   default=500)
    ap.add_argument("--sim-budgets",          type=str,   default="8,32,128")
    ap.add_argument("--eval-games",           type=int,   default=100)
    ap.add_argument("--eval-every",           type=int,   default=50)
    ap.add_argument("--train-steps",          type=int,   default=2)
    ap.add_argument("--batch-size",           type=int,   default=64)
    ap.add_argument("--replay-size",          type=int,   default=5000)
    ap.add_argument("--h2h-games",            type=int,   default=200)
    ap.add_argument("--seed",                 type=int,   default=42)
    ap.add_argument("--minimax-depth",        type=int,   default=2)
    ap.add_argument("--minimax-eps",          type=float, default=0.3)
    ap.add_argument("--out",                  type=str,   default="demo/png/connect4_train_curves.png")
    ap.add_argument("--max-nodes",            type=int,   default=500)
    ap.add_argument("--max-considered-actions", type=int, default=7)
    ap.add_argument("--entropy-min-k",        type=int,   default=7)
    args = ap.parse_args()

    sim_budgets = [int(x) for x in args.sim_budgets.split(",")]
    logic       = Connect4Logic()

    dummy = np.zeros((ROWS, COLS), dtype=np.int8)
    _fast_step(dummy, 3, 1); _valid_mask(dummy, 1)
    print("Compiling minimax kernel...")
    minimax_action(dummy.copy(), 1, 1)
    print("Numba kernels ready.\n")
    print(f"Connect 4 — Gumbel vs PUCT  |  budgets={sim_budgets}")
    print(f"  episodes={args.games}  eval_every={args.eval_every}")
    print(f"  minimax depth={args.minimax_depth}  ε={args.minimax_eps}\n")

    results, nets = {}, {}
    for sims in sim_budgets:
        for algo in ["gumbel", "puct"]:
            print(f"\n=== {algo.upper()} · {sims} sims ===")
            net, curve            = train_system(algo, args, logic, sims)
            results[(algo, sims)] = curve
            nets[(algo, sims)]    = net

    print(f"\n=== Head-to-head (Gumbel-trained vs PUCT-trained, n={args.h2h_games} games each) ===")
    for sims in sim_budgets:
        gn, pn = nets[("gumbel", sims)], nets[("puct", sims)]
        gn.eval(); pn.eval()
        with torch.no_grad():
            gwr, g_wins = head_to_head(gn, pn, logic, args.h2h_games, sims, args)
        p_wins = args.h2h_games - g_wins
        g_lo, g_hi = wilson_ci(g_wins, args.h2h_games)
        p_lo, p_hi = wilson_ci(p_wins, args.h2h_games)
        print(f"  {sims:3d} sims:  Gumbel {gwr:.1f}% [{g_lo:.0f}%-{g_hi:.0f}%]  "
              f"PUCT {100-gwr:.1f}% [{p_lo:.0f}%-{p_hi:.0f}%]")

    plot_results(results, sim_budgets, args.minimax_eps, args.out)


if __name__ == "__main__":
    main()
