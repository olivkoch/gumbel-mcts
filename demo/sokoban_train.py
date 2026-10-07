"""
demo/sokoban_train.py — Self-play training: Gumbel vs PUCT learning curves.

Compares two complete pipelines end-to-end on official Microban Sokoban puzzles:
  System A: Gumbel self-play → Gumbel-trained model → Gumbel evaluation
  System B: PUCT self-play   → PUCT-trained model   → PUCT evaluation

Training set: Microban #1, #20, #21 (each episode randomly picks one).
Eval puzzle:  Microban #19 (held out — the model never trains on it).

Metric: success rate on the held-out eval puzzle vs self-play episodes consumed.

Usage
-----
    uv run python demo/sokoban_train.py
    uv run python demo/sokoban_train.py --sims 64 --iters 40 --n-eval 30
"""

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gumbel_mcts import PUCT, GumbelDense

# ── Constants ──────────────────────────────────────────────────────────────────
N_BOXES     = 2    # all chosen puzzles have exactly 2 boxes
NUM_ACTIONS = N_BOXES * 4 + 1  # N_BOXES×4 macro-pushes (box_idx*4+dir) + 1 pass
MAX_H = 10         # CNN input size (all puzzles padded to this)
MAX_W = 10

# Action deltas: up(-y) right(+x) down(+y) left(-x)
ADX = np.array([ 0, 1,  0, -1], dtype=np.int8)
ADY = np.array([-1, 0,  1,  0], dtype=np.int8)


# ── Player reachability BFS (module-level so closure @njit functions can call it)

@njit(cache=False)
def _bfs_reach(px, py, tx, ty, walls, gh, gw, boxes_x, boxes_y):
    """Return True if player at (px,py) can walk to (tx,ty) without crossing boxes."""
    if px == tx and py == ty:
        return True
    max_cells = gh * gw
    visited = np.zeros(max_cells, dtype=np.bool_)
    qx = np.empty(max_cells, dtype=np.int32)
    qy = np.empty(max_cells, dtype=np.int32)
    head = np.int32(0); tail = np.int32(0)
    visited[py * gw + px] = True
    qx[tail] = np.int32(px); qy[tail] = np.int32(py); tail += np.int32(1)
    while head < tail:
        cx = qx[head]; cy = qy[head]; head += np.int32(1)
        for d in range(4):
            nx = cx + np.int32(ADX[d]); ny = cy + np.int32(ADY[d])
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            if walls[ny, nx]:
                continue
            nidx = ny * gw + nx
            if visited[nidx]:
                continue
            blocked = False
            for i in range(N_BOXES):
                if boxes_x[i] == nx and boxes_y[i] == ny:
                    blocked = True
                    break
            if blocked:
                continue
            if nx == tx and ny == ty:
                return True
            visited[nidx] = True
            qx[tail] = nx; qy[tail] = ny; tail += np.int32(1)
    return False


# ── Microban puzzle texts (David W. Skinner, public domain) ───────────────────
# Training set: 3 different 2-box puzzles, varied layouts.
# Eval puzzle:  held-out 2-box puzzle, never seen during training.

MICROBAN_1 = """\
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####"""

MICROBAN_20 = """\
#######
#     ###
#  @$$..#
#### ## #
  #     #
  #  ####
  #  #
  ####"""

MICROBAN_21 = """\
####
#  ####
# . . #
# $$#@#
##    #
 ######"""

MICROBAN_19 = """\
########
#   .. #
#  @$$ #
##### ##
   #  #
   #  #
   #  #
   ####"""

TRAIN_TEXTS = [MICROBAN_1, MICROBAN_20, MICROBAN_21]
EVAL_TEXT   = MICROBAN_19


# ── Puzzle dataclass and parser ────────────────────────────────────────────────

@dataclass
class SokobanPuzzle:
    walls:   np.ndarray   # (H, W) int8
    targets: np.ndarray   # (N_BOXES, 2) int8
    start:   np.ndarray   # (2+2*N_BOXES,) int8
    label:   str = ""

    def __post_init__(self):
        self.grid_h, self.grid_w = self.walls.shape
        # Precomputed padded CNN channels (MAX_H × MAX_W)
        wt = torch.zeros(MAX_H, MAX_W, dtype=torch.float32)
        wt[:self.grid_h, :self.grid_w] = torch.tensor(
            self.walls, dtype=torch.float32)
        wt[self.grid_h:, :] = 1.0
        wt[:, self.grid_w:] = 1.0
        self._walls_ch = wt

        tgt = torch.zeros(MAX_H, MAX_W, dtype=torch.float32)
        for t in self.targets:
            tgt[int(t[1]), int(t[0])] = 1.0
        self._targets_ch = tgt


def parse_puzzle(text: str, label: str = "") -> SokobanPuzzle:
    lines = text.strip("\n").split("\n")
    width = max(len(l) for l in lines)
    lines = [l.ljust(width) for l in lines]
    height = len(lines)

    # Flood-fill from border to mark void (outside) spaces as walls
    outside = [[False] * width for _ in range(height)]
    q: deque = deque()
    for r in range(height):
        for c in range(width):
            if lines[r][c] == " " and not outside[r][c]:
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    outside[r][c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if lines[nr][nc] == " " and not outside[nr][nc]:
                    outside[nr][nc] = True
                    q.append((nr, nc))

    walls   = np.zeros((height, width), dtype=np.int8)
    player, boxes, targets = None, [], []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#" or (ch == " " and outside[r][c]):
                walls[r, c] = 1
            if ch in "@+":
                player = [c, r]
            if ch in "$*":
                boxes.append([c, r])
            if ch in ".*+":
                targets.append([c, r])

    assert player is not None, f"No player in '{label}'"
    assert len(boxes)   == N_BOXES, f"{N_BOXES} boxes expected, got {len(boxes)} in '{label}'"
    assert len(targets) == N_BOXES, f"{N_BOXES} targets expected, got {len(targets)} in '{label}'"

    start = np.array(player + [v for b in boxes for v in b], dtype=np.int8)
    tgt   = np.array(targets, dtype=np.int8)
    return SokobanPuzzle(walls=walls, targets=tgt, start=start, label=label)


# ── Numba function factory ─────────────────────────────────────────────────────
# gumbel_mcts passes logic.fast_step / logic.get_valid_mask as arguments to
# Numba kernels, so they must be @njit-compiled free functions.
# We create one (fast_step, get_valid_mask) pair per puzzle via closures.

def _make_game_fns(walls: np.ndarray, targets: np.ndarray,
                   grid_h: int, grid_w: int):
    _w  = np.ascontiguousarray(walls,   dtype=np.int8)
    _t  = np.ascontiguousarray(targets, dtype=np.int8)
    _gh = int(grid_h)
    _gw = int(grid_w)

    @njit(cache=False)
    def fast_step(board, action, player):
        # Player 2 always passes (2-player alternation trick for single-player game).
        if player == np.int8(2) or action == np.int32(N_BOXES * 4):
            return 0.0, np.int32(1), False, board

        # Decode macro-action: action = box_idx * 4 + dir_idx
        box_idx = np.int32(action) // np.int32(4)
        dir_idx = np.int32(action) % np.int32(4)

        bx = np.int32(board[2 + 2 * box_idx])
        by = np.int32(board[3 + 2 * box_idx])
        px = np.int32(board[0]); py = np.int32(board[1])
        dx = np.int32(ADX[dir_idx]); dy = np.int32(ADY[dir_idx])

        # Target square (where box lands after push)
        tx = bx + dx; ty = by + dy
        if ty < 0 or ty >= _gh or tx < 0 or tx >= _gw:
            return 0.0, np.int32(1), False, board
        if _w[ty, tx]:
            return 0.0, np.int32(1), False, board

        # Target must not be occupied by another box
        for i in range(N_BOXES):
            if i != box_idx:
                if np.int32(board[2 + 2*i]) == tx and np.int32(board[3 + 2*i]) == ty:
                    return 0.0, np.int32(1), False, board

        # Pushing square: player must stand here to push the box
        psx = bx - dx; psy = by - dy
        if psy < 0 or psy >= _gh or psx < 0 or psx >= _gw:
            return 0.0, np.int32(1), False, board
        if _w[psy, psx]:
            return 0.0, np.int32(1), False, board

        # Collect current box positions for BFS (boxes block player movement)
        boxes_x = np.empty(N_BOXES, dtype=np.int32)
        boxes_y = np.empty(N_BOXES, dtype=np.int32)
        for i in range(N_BOXES):
            boxes_x[i] = np.int32(board[2 + 2*i])
            boxes_y[i] = np.int32(board[3 + 2*i])

        # BFS: can player walk to the pushing square?
        if not _bfs_reach(px, py, psx, psy, _w, _gh, _gw, boxes_x, boxes_y):
            return 0.0, np.int32(1), False, board

        # Execute: player moves to old box position, box slides to target
        board[0] = np.int8(bx); board[1] = np.int8(by)
        board[2 + 2 * box_idx] = np.int8(tx)
        board[3 + 2 * box_idx] = np.int8(ty)

        all_on = True
        for i in range(N_BOXES):
            found = False
            for j in range(N_BOXES):
                if (np.int32(board[2 + 2*i]) == np.int32(_t[j, 0]) and
                        np.int32(board[3 + 2*i]) == np.int32(_t[j, 1])):
                    found = True
                    break
            if not found:
                all_on = False
                break

        if all_on:
            return 1.0, np.int32(1), True, board
        return 0.0, np.int32(2), False, board

    @njit(cache=False)
    def get_valid_mask(board, player):
        mask = np.zeros(N_BOXES * 4 + 1, dtype=np.float32)
        if player == np.int8(2):
            mask[N_BOXES * 4] = 1.0
            return mask

        px = np.int32(board[0]); py = np.int32(board[1])
        boxes_x = np.empty(N_BOXES, dtype=np.int32)
        boxes_y = np.empty(N_BOXES, dtype=np.int32)
        for i in range(N_BOXES):
            boxes_x[i] = np.int32(board[2 + 2*i])
            boxes_y[i] = np.int32(board[3 + 2*i])

        for box_idx in range(N_BOXES):
            bx = boxes_x[box_idx]; by = boxes_y[box_idx]
            for dir_idx in range(4):
                dx = np.int32(ADX[dir_idx]); dy = np.int32(ADY[dir_idx])
                tx = bx + dx; ty = by + dy
                if ty < 0 or ty >= _gh or tx < 0 or tx >= _gw:
                    continue
                if _w[ty, tx]:
                    continue
                ok = True
                for i in range(N_BOXES):
                    if i != box_idx and boxes_x[i] == tx and boxes_y[i] == ty:
                        ok = False; break
                if not ok:
                    continue
                psx = bx - dx; psy = by - dy
                if psy < 0 or psy >= _gh or psx < 0 or psx >= _gw:
                    continue
                if _w[psy, psx]:
                    continue
                if _bfs_reach(px, py, psx, psy, _w, _gh, _gw, boxes_x, boxes_y):
                    mask[box_idx * 4 + dir_idx] = 1.0

        return mask

    return fast_step, get_valid_mask


# ── BFS solver ────────────────────────────────────────────────────────────────

def solve_bfs(game: "SokobanGame", max_states: int = 1_000_000):
    """Return a shortest action sequence for the puzzle, or None if unsolvable."""
    def canonical(board):
        boxes = sorted((int(board[2+2*i]), int(board[3+2*i])) for i in range(N_BOXES))
        return (int(board[0]), int(board[1])) + tuple(v for b in boxes for v in b)

    start = game.get_initial_board()
    s0    = canonical(start)
    visited: dict = {s0: (None, -1)}   # canon -> (parent_canon, action)
    queue  = deque([(start.copy(), s0)])

    while queue and len(visited) <= max_states:
        board, s = queue.popleft()
        for action in range(NUM_ACTIONS - 1):          # 0-3, skip pass
            nb       = board.copy()
            reward, _, done, nb = game.fast_step(nb, action, 1)
            ns       = canonical(nb)
            if ns == s:                                 # invalid move (board unchanged)
                continue
            if ns in visited:
                continue
            visited[ns] = (s, action)
            if done:                                    # solved — reconstruct path
                path: list = []
                cur = ns
                while visited[cur][0] is not None:
                    parent, act = visited[cur]
                    path.append(act)
                    cur = parent
                return list(reversed(path))
            queue.append((nb, ns))
    return None


def bfs_episode(game: "SokobanGame", solution: list):
    """Turn a BFS solution into training data, including off-path negatives.

    On-path states get GRADED values: step k gets value (k+1)/total in (0,1].
    This ensures Q(forward step) > Q(undo step) at every BFS position, because
    undo leads to step-(k-1) with value k/total < (k+1)/total.
    Off-path wrong states get value=-0.5. BFS path boards are excluded from
    off-path negatives to prevent contaminating the graded on-path signal.

    Each tuple stores (board, policy, value, walls_ch, targets_ch) so the
    encoding context travels with the data and never relies on index lookup.
    """
    walls_ch   = game.puzzle._walls_ch
    targets_ch = game.puzzle._targets_ch

    # Precompute canonical forms of all BFS path boards to avoid contamination.
    bfs_path: set = set()
    b = game.get_initial_board()
    bfs_path.add(b.tobytes())
    for a in solution:
        _, _, _, b = game.fast_step(b, a, 1)
        bfs_path.add(b.tobytes())

    board  = game.get_initial_board()
    data: list = []
    unif = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    total = len(solution)  # needed for graded values

    for k, action in enumerate(solution):
        pol          = np.full(NUM_ACTIONS, 0.05 / (NUM_ACTIONS - 1), dtype=np.float32)
        pol[action] += 0.95
        pol         /= pol.sum()
        # Graded value: step k board gets (k+1)/total in (0,1].
        # This ensures Q(forward) > Q(undo) at every BFS step because the undo
        # board is step-(k-1) with value k/total < (k+1)/total.
        on_path_value = float(k + 1) / float(total)
        data.append((board.copy(), pol, on_path_value, walls_ch, targets_ch))

        for wrong in range(NUM_ACTIONS - 1):   # skip pass
            if wrong == action:
                continue
            nb = board.copy()
            _, _, _, nb = game.fast_step(nb, wrong, 1)
            if np.array_equal(nb, board):
                continue
            if nb.tobytes() in bfs_path:
                continue   # undo action landed on a BFS state — skip
            data.append((nb.copy(), unif.copy(), -0.5, walls_ch, targets_ch))
            # 2-hop: one more step from the wrong board.
            for wrong2 in range(NUM_ACTIONS - 1):
                nb2 = nb.copy()
                _, _, _, nb2 = game.fast_step(nb2, wrong2, 1)
                if not np.array_equal(nb2, nb) and nb2.tobytes() not in bfs_path:
                    data.append((nb2.copy(), unif.copy(), -0.5, walls_ch, targets_ch))
                    break

        _, _, done, board = game.fast_step(board, action, 1)
        if done:
            break
    return data


# ── Game logic class ──────────────────────────────────────────────────────────

class SokobanGame:
    """gumbel_mcts-compatible game logic bound to one Sokoban puzzle."""

    NUM_ACTIONS     = NUM_ACTIONS
    BOARD_SHAPE     = (2 + 2 * N_BOXES,)
    MAX_MOVES       = 60    # macro-pushes: solutions are 5-16 deep, 30 steps is plenty
    MAX_LEGAL_MOVES = N_BOXES * 4
    PLAYER_1        = 1
    PLAYER_2        = 2

    def __init__(self, puzzle: SokobanPuzzle):
        self.puzzle = puzzle
        self.fast_step, self.get_valid_mask = _make_game_fns(
            puzzle.walls, puzzle.targets, puzzle.grid_h, puzzle.grid_w)

    def get_initial_board(self) -> np.ndarray:
        return self.puzzle.start.copy()


# ── Board encoder ─────────────────────────────────────────────────────────────

def encode_boards(boards_np: np.ndarray,
                  walls_ch: torch.Tensor,
                  targets_ch: torch.Tensor) -> torch.Tensor:
    """boards_np (B, 6) → (B, C, MAX_H, MAX_W)  with C = N_BOXES+3."""
    B = len(boards_np)
    C = N_BOXES + 3   # walls | player | box0 | box1 | targets
    x = torch.zeros(B, C, MAX_H, MAX_W, dtype=torch.float32)
    x[:, 0]         = walls_ch
    x[:, N_BOXES+2] = targets_ch
    for i in range(B):
        b = boards_np[i]
        x[i, 1, int(b[1]), int(b[0])] = 1.0
        for j in range(N_BOXES):
            x[i, 2+j, int(b[3+2*j]), int(b[2+2*j])] = 1.0
    return x


# ── Neural network ────────────────────────────────────────────────────────────

class SokobanNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = N_BOXES + 3
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * MAX_H * MAX_W, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, NUM_ACTIONS)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return F.softmax(self.policy_head(h), dim=-1), torch.tanh(self.value_head(h))


# ── Heuristic model (warm-start only) ─────────────────────────────────────────

def _corner_dead(bx, by, walls, h, w):
    wu = walls[by-1, bx] if by > 0   else 1
    wd = walls[by+1, bx] if by < h-1 else 1
    wl = walls[by, bx-1] if bx > 0   else 1
    wr = walls[by, bx+1] if bx < w-1 else 1
    return bool((wu and wl) or (wu and wr) or (wd and wl) or (wd and wr))


class HeuristicModel:
    def __init__(self, game: SokobanGame):
        self.logic  = game
        self._tgts  = game.puzzle.targets
        self._walls = game.puzzle.walls
        self._h     = game.puzzle.grid_h
        self._w     = game.puzzle.grid_w

    def forward_for_mcts(self, batch):
        B       = batch["boards"].shape[0]
        boards  = batch["boards"].float()
        players = batch["current_player"]

        tx0, ty0 = float(self._tgts[0, 0]), float(self._tgts[0, 1])
        tx1, ty1 = float(self._tgts[1, 0]), float(self._tgts[1, 1])
        bx0 = boards[:, 2]; by0 = boards[:, 3]
        bx1 = boards[:, 4]; by1 = boards[:, 5]
        d01  = (bx0-tx0).abs()+(by0-ty0).abs()+(bx1-tx1).abs()+(by1-ty1).abs()
        d10  = (bx0-tx1).abs()+(by0-ty1).abs()+(bx1-tx0).abs()+(by1-ty0).abs()
        dist = torch.minimum(d01, d10)

        policy = torch.zeros(B, NUM_ACTIONS)
        value  = torch.zeros(B, 1)
        p1 = (players == 1).nonzero(as_tuple=True)[0]
        p2 = (players == 2).nonzero(as_tuple=True)[0]

        if len(p1):
            policy[p1, :4] = 0.25
            for k in p1.tolist():
                dead = any(
                    _corner_dead(int(boards[k, 2+2*i]), int(boards[k, 3+2*i]),
                                 self._walls, self._h, self._w)
                    for i in range(N_BOXES))
                value[k] = -1.0 if dead else -torch.tanh(dist[k] / 6.0)

        if len(p2):
            policy[p2, N_BOXES * 4] = 1.0
            for k in p2.tolist():
                dead = any(
                    _corner_dead(int(boards[k, 2+2*i]), int(boards[k, 3+2*i]),
                                 self._walls, self._h, self._w)
                    for i in range(N_BOXES))
                value[k] = 1.0 if dead else torch.tanh(dist[k] / 6.0)

        return {"policy": policy, "value": value}


# ── Neural model wrapper ──────────────────────────────────────────────────────

class NeuralModel:
    def __init__(self, net: SokobanNet, game: SokobanGame):
        self.net    = net
        self.logic  = game        # required by gumbel_mcts internals
        self.puzzle = game.puzzle

    def forward_for_mcts(self, batch):
        boards  = batch["boards"].numpy()
        players = batch["current_player"]
        B       = len(boards)

        x = encode_boards(boards, self.puzzle._walls_ch, self.puzzle._targets_ch)
        with torch.no_grad():
            policy, value = self.net(x)

        vm = batch.get("valid_mask")
        if vm is not None:
            policy = policy * vm
            policy = policy / policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        for i in range(B):
            if players[i].item() == 2:
                policy[i].zero_(); policy[i, N_BOXES * 4] = 1.0
                value[i] = -value[i]

        return {"policy": policy, "value": value}


# ── Episode runner ─────────────────────────────────────────────────────────────

def _heuristic_value(board: np.ndarray, targets: np.ndarray) -> float:
    """Min-assignment Manhattan distance → value in (-1, 0]."""
    bx0, by0 = int(board[2]), int(board[3])
    bx1, by1 = int(board[4]), int(board[5])
    tx0, ty0 = int(targets[0, 0]), int(targets[0, 1])
    tx1, ty1 = int(targets[1, 0]), int(targets[1, 1])
    d01 = abs(bx0-tx0) + abs(by0-ty0) + abs(bx1-tx1) + abs(by1-ty1)
    d10 = abs(bx0-tx1) + abs(by0-ty1) + abs(bx1-tx0) + abs(by1-ty0)
    return -float(np.tanh(min(d01, d10) / 6.0))


def _partial_value(board: np.ndarray, targets: np.ndarray) -> float:
    """Value target for non-terminal states: pure distance, always in (-1, 0].

    Using the box-placement floor (crossing 0 when first box lands) creates a
    1-unit jump that maps to a 50-unit swing in Gumbel's sigma*Q score,
    overwhelming policy logits and destroying sequential halving.  Pure
    distance keeps action-to-action Q differences small (~0.05) so Gumbel
    noise and policy priors remain meaningful.
    """
    return _heuristic_value(board, targets)


def collect_episode(algo: str, sims: int, game: SokobanGame, model):
    """Run one self-play episode.  Returns (data, success).

    Win:       all states → +1.0
    Deadlock:  all states → -1.0  (no valid push exists; position is unwinnable)
    Timeout:   each state → _partial_value(state) = -tanh(dist/6) in (-1, 0]
    """
    board     = game.get_initial_board()
    states, policies = [], []
    max_nodes = max(sims * 10 + 200, 600)
    deadlocked = False

    for _ in range(game.MAX_MOVES // 2):
        # Deadlock: no valid macro-push reachable from current position
        vm = game.get_valid_mask(board, np.int8(1))
        if vm[:-1].sum() == 0:
            deadlocked = True
            break

        Cls  = GumbelDense if algo == "gumbel" else PUCT
        tree = Cls(n_games=1, max_nodes=max_nodes, logic=game, device="cpu")
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=sims)

        # Action selection: visit counts maintain exploration diversity
        visits, _ = tree.get_all_root_data(n_active=1)
        v        = visits[0].astype(np.float64)
        sample_pol = v / v.sum() if v.sum() > 1e-9 else np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        sample_pol = sample_pol * vm
        sample_pol = sample_pol / sample_pol.sum() if sample_pol.sum() > 1e-9 else vm / vm.sum()

        # Training target: Gumbel uses the Q-completed improved policy from the paper;
        # PUCT uses the same visit distribution used for action selection.
        if algo == "gumbel":
            train_pol = tree.get_improved_policy(n_active=1)[0].astype(np.float64)
            train_pol = train_pol * vm
            train_pol = train_pol / train_pol.sum() if train_pol.sum() > 1e-9 else vm / vm.sum()
        else:
            train_pol = sample_pol.copy()

        states.append(board.copy())
        policies.append(train_pol)

        action = int(np.random.choice(NUM_ACTIONS, p=sample_pol))
        _, _, done, board = game.fast_step(board, action, 1)
        if done:
            wc = game.puzzle._walls_ch
            tc = game.puzzle._targets_ch
            return [(s, p, 1.0, wc, tc) for s, p in zip(states, policies)], True

    if not states:
        return [], False
    wc = game.puzzle._walls_ch
    tc = game.puzzle._targets_ch
    if deadlocked:
        return [(s, p, -1.0, wc, tc) for s, p in zip(states, policies)], False
    targets = game.puzzle.targets
    return [(s, p, _partial_value(s, targets), wc, tc) for s, p in zip(states, policies)], False


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(net: SokobanNet, optimizer, replay_buffer: list,
               batch_size: int = 64, value_weight: float = 1.0,
               bfs_buffer: list = None):
    # Sample 50% from the fixed BFS buffer (correct policy + value=1) and
    # 50% from the rolling self-play buffer.  This prevents self-play's
    # uniform-visit policies from overwriting the BFS policy signal.
    if bfs_buffer and len(replay_buffer) >= 8:
        half = batch_size // 2
        n_bfs  = min(half, len(bfs_buffer))
        n_play = min(batch_size - n_bfs, len(replay_buffer))
        batch  = random.sample(bfs_buffer, n_bfs) + random.sample(replay_buffer, n_play)
    else:
        n     = min(batch_size, len(replay_buffer))
        batch = random.sample(replay_buffer, n)
    boards, mcts_pols, outcomes, walls_chs, targets_chs = zip(*batch)

    xs = [encode_boards(np.array([b]), wc, tc)[0]
          for b, wc, tc in zip(boards, walls_chs, targets_chs)]
    x  = torch.stack(xs)
    tp = torch.tensor(np.stack(mcts_pols), dtype=torch.float32)
    tv = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)

    net.train()
    pred_p, pred_v = net(x)
    policy_loss = -(tp * (pred_p + 1e-8).log()).sum(-1).mean()
    value_loss  = F.mse_loss(pred_v, tv)
    loss        = policy_loss + value_weight * value_loss

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item()}


# ── Evaluation on held-out puzzle ─────────────────────────────────────────────

def evaluate_system(algo: str, sims: int, eval_game: SokobanGame,
                    net: SokobanNet, n_eval: int, seed: int) -> float:
    net.eval()
    model = NeuralModel(net, eval_game)
    wins  = 0
    for ep in range(n_eval):
        np.random.seed(seed + ep * 137); torch.manual_seed(seed + ep * 137)
        _, success = collect_episode(algo, sims, eval_game, model)
        wins += success
    return wins / n_eval * 100.0


# ── Main training loop ────────────────────────────────────────────────────────

def train_system(algo: str, args, train_games: List[SokobanGame],
                 eval_game: SokobanGame, seed: int, group: str):
    print(f"\n=== System: {algo.upper()} ===")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    run = wandb.init(
        entity="mistral-ai",
        project="mcts",
        name=algo,
        group=group,
        config={
            "algo":             algo,
            "sims":             args.sims,
            "sims_warmup":      args.sims_warmup,
            "sims_eval":        args.sims_eval,
            "warmup":           args.warmup,
            "iters":            args.iters,
            "episodes_per_iter": args.episodes_per_iter,
            "train_steps":      args.train_steps,
            "eval_every":       args.eval_every,
            "n_eval":           args.n_eval,
            "replay_size":      args.replay_size,
            "seed":             seed,
            "min_wins":         args.min_wins,
            "train_puzzles":    "Microban #1, #20, #21",
            "eval_puzzle":      "Microban #19",
        },
        reinit=True,
    )

    net           = SokobanNet()
    optimizer     = torch.optim.Adam(net.parameters(), lr=3e-4)
    replay: list  = []    # rolling self-play buffer (trimmed to replay_size)
    bfs_buf: list = []    # permanent BFS buffer (never overwritten by self-play)
    curve:        list  = []   # (episodes, eval_success_rate)
    train_curve:  list  = []   # (episodes, rolling_train_win_rate)
    episodes      = 0
    train_step_n  = 0
    wins_window:  list  = []   # recent per-iter win rates for smoothed train curve

    # ── Warm-start with BFS solutions ─────────────────────────────────────────
    print(f"  Warm-start: solving training puzzles with BFS ...")
    solutions = {}
    for pidx, game in enumerate(train_games):
        sol = solve_bfs(game)
        if sol is None:
            print(f"    WARNING: BFS could not solve {game.puzzle.label}")
        else:
            solutions[pidx] = sol
            print(f"    {game.puzzle.label}: {len(sol)}-move solution found")

    ep_per_puzzle = args.warmup // max(len(solutions), 1)
    wins_warm = 0
    for pidx, sol in solutions.items():
        for _ in range(ep_per_puzzle):
            data = bfs_episode(train_games[pidx], sol)
            bfs_buf.extend(data)   # goes into permanent BFS buffer
            wins_warm += 1
    print(f"  Warm-start: {wins_warm} BFS episodes, bfs_buf={len(bfs_buf)}")
    run.log({"warm_start/wins": wins_warm,
             "warm_start/total": args.warmup,
             "warm_start/win_rate": 1.0}, step=0)

    # Pre-train on BFS data alone so the policy/value head starts correctly.
    pretrain_steps = getattr(args, "pretrain_steps", 1000)
    for _ in range(pretrain_steps):
        if len(bfs_buf) >= 16:
            losses = train_step(net, optimizer, bfs_buf,
                                batch_size=min(64, len(bfs_buf)),
                                value_weight=2.0)
            run.log({"train/loss":        losses["loss"],
                     "train/policy_loss": losses["policy_loss"],
                     "train/value_loss":  losses["value_loss"],
                     "train/step":        train_step_n}, step=train_step_n)
            train_step_n += 1

    sr = evaluate_system(algo, args.sims_eval, eval_game, net,
                         args.n_eval, seed=9_000_000 + seed) if args.n_eval > 0 else 0.0
    curve.append((0, sr))
    run.log({"eval/success_rate": sr, "train/episodes": 0}, step=train_step_n)
    print(f"  [iter   0] baseline after warm-start: {sr:.1f}%")

    # ── Neural self-play ──────────────────────────────────────────────────────
    wins_total = wins_warm
    for it in range(1, args.iters + 1):
        net.train()
        wins_iter = 0

        for ep in range(args.episodes_per_iter):
            pidx = random.randrange(len(train_games))
            game = train_games[pidx]
            np.random.seed(seed + 10_000 + it * 1000 + ep)
            torch.manual_seed(seed + 10_000 + it * 1000 + ep)
            model = NeuralModel(net, game)
            data, ok = collect_episode(algo, args.sims, game, model)
            replay.extend(data)
            episodes += 1
            wins_iter += ok
            wins_total += ok
        replay = replay[-args.replay_size:]

        value_weight = 2.0

        iter_losses = []
        for _ in range(args.train_steps):
            if len(replay) >= 8:
                losses = train_step(net, optimizer, replay,
                                    batch_size=min(64, len(replay)),
                                    value_weight=value_weight,
                                    bfs_buffer=bfs_buf)
                iter_losses.append(losses)
                run.log({"train/loss":         losses["loss"],
                         "train/policy_loss":  losses["policy_loss"],
                         "train/value_loss":   losses["value_loss"],
                         "train/value_weight": value_weight,
                         "train/step":         train_step_n}, step=train_step_n)
                train_step_n += 1

        run.log({
            "train/episodes":        episodes,
            "train/wins_this_iter":  wins_iter,
            "train/win_rate_iter":   wins_iter / max(args.episodes_per_iter, 1),
            "train/buffer_size":     len(replay),
            "train/iter":            it,
        }, step=train_step_n)

        wins_window.append(wins_iter / max(args.episodes_per_iter, 1))

        if it % args.eval_every == 0:
            if args.n_eval > 0:
                sr = evaluate_system(algo, args.sims_eval, eval_game, net,
                                     args.n_eval, seed=9_000_000 + seed)
            else:
                sr = 0.0   # skip eval when n_eval=0 (sweep mode)
            curve.append((episodes, sr))
            # Smoothed training win rate over the last eval_every iters
            train_wr = sum(wins_window[-args.eval_every:]) / max(len(wins_window[-args.eval_every:]), 1) * 100.0
            train_curve.append((episodes, train_wr))
            run.log({"eval/success_rate": sr,
                     "train/episodes":    episodes,
                     "train/win_rate_smooth": train_wr}, step=train_step_n)
            print(f"  [iter {it:3d}] ep={episodes:4d}  "
                  f"train-wins={wins_iter}/{args.episodes_per_iter}  "
                  f"train-wr={train_wr:.0f}%  eval={sr:5.1f}%  buf={len(replay)}")

    run.finish()
    return curve, train_curve


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_curves(gumbel_curve, puct_curve, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve, color, marker in [
        ("Gumbel", gumbel_curve, "#26A69A", "s"),
        ("PUCT",   puct_curve,   "#5C6BC0", "o"),
    ]:
        ep = [c[0] for c in curve]
        sr = [c[1] for c in curve]
        ax.plot(ep, sr, f"{marker}-", color=color, label=label,
                lw=2.5, ms=7, mfc="white", mew=2)

    ax.set_xlabel("Self-play training episodes", fontsize=11)
    ax.set_ylabel("Success rate on Microban #19 (%)", fontsize=11)
    ax.set_title("Gumbel vs PUCT: self-play learning curves\n"
                 "(train: Microban #1/#20/#21  —  eval: Microban #19, held-out)",
                 fontsize=11, fontweight="bold")
    ax.axhline(50, color="#BDBDBD", ls="--", lw=1)
    ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims",              type=int, default=64)
    ap.add_argument("--sims-warmup",       type=int, default=256,
                    help="sims per move during heuristic warm-start (higher = more wins)")
    ap.add_argument("--sims-eval",         type=int, default=128)
    ap.add_argument("--warmup",            type=int, default=60)
    ap.add_argument("--iters",             type=int, default=30)
    ap.add_argument("--episodes-per-iter", type=int, default=5)
    ap.add_argument("--train-steps",       type=int, default=20)
    ap.add_argument("--eval-every",        type=int, default=5)
    ap.add_argument("--n-eval",            type=int, default=40)
    ap.add_argument("--replay-size",       type=int, default=8000)
    ap.add_argument("--min-wins",          type=int, default=5,
                    help="min cumulative wins before value head trains")
    ap.add_argument("--seed",              type=int, default=0)
    ap.add_argument("--out",               type=str,
                    default="demo/png/sokoban_train_plot.png")
    args = ap.parse_args()

    print("Sokoban — Gumbel vs PUCT self-play learning curves")
    print("  Train: Microban #1, #20, #21  |  Eval (held-out): Microban #19")
    print(f"  sims={args.sims} (train)  sims_eval={args.sims_eval}  warmup=BFS")
    print(f"  warmup={args.warmup}  iters={args.iters}  ep/iter={args.episodes_per_iter}")
    print("\nCompiling Numba game kernels for 4 puzzles ...")

    train_puzzles = [parse_puzzle(t, f"Microban #{n}")
                     for t, n in zip(TRAIN_TEXTS, [1, 20, 21])]
    eval_puzzle   = parse_puzzle(EVAL_TEXT, "Microban #19")
    train_games   = [SokobanGame(p) for p in train_puzzles]
    eval_game     = SokobanGame(eval_puzzle)

    # Warm-up Numba JIT so training timings are clean
    dummy = np.zeros(2 + 2 * N_BOXES, dtype=np.int8)
    for g in train_games + [eval_game]:
        g.fast_step(dummy.copy(), 0, 1)
        g.get_valid_mask(dummy, 1)
    print("Kernels ready.\n")

    # Runs in the same group so they appear together in the wandb UI
    group = f"sokoban-seed{args.seed}"

    gumbel_curve, _ = train_system("gumbel", args, train_games, eval_game,
                                   seed=args.seed,        group=group)
    puct_curve,   _ = train_system("puct",   args, train_games, eval_game,
                                   seed=args.seed + 1000, group=group)

    plot_curves(gumbel_curve, puct_curve, args.out)

    # Upload the comparison plot to a summary run
    with wandb.init(entity="mistral-ai", project="mcts", name="summary", group=group, reinit=True):
        wandb.log({"learning_curves": wandb.Image(args.out)})


if __name__ == "__main__":
    main()
