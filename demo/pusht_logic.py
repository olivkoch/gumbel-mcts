"""
PushT wrapped as a GameLogic for the gumbel_mcts library.

The physics sim (pymunk) can't be numba-compiled, so this module provides:
  - PushTLogic: GameLogic protocol with a Python-callable fast_step
  - PushTModel: MCTSModel that returns diffusion prior + zero value
  - PythonPUCT / PythonGumbelDense: subclasses that use pure-Python kernels
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import gymnasium as gym
import gym_pusht  # noqa: F401

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_PUSH_DIRS = 8
NUM_ACTIONS   = 9       # 0-7 push directions + 8 = no-op
APPROACH_DIST = 80
PUSH_DEPTH    = 60
N_APPROACH    = 12
N_PUSH        = 18
N_SETTLE      = 12

# Adaptive push: scale down when block is close to goal
NEAR_DIST_THRESHOLD = 80.0   # keypoint distance below which pushes get shorter
NEAR_PUSH_SCALE     = 0.35   # multiply step counts by this when near goal

WORKSPACE_LO, WORKSPACE_HI = 0.0, 512.0

# ── Goal geometry (for adaptive push + value) ────────────────────────────────

LOCAL_VERTS = np.array([
    [-60, 0], [60, 0], [60, 30], [-60, 30],
    [-15, 30], [15, 30], [15, 120], [-15, 120],
], dtype=np.float64)

GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])

def _goal_keypoints():
    gx, gy, ga = GOAL_POSE
    c, s = np.cos(ga), np.sin(ga)
    R = np.array([[c, -s], [s, c]])
    return (R @ LOCAL_VERTS.T).T + np.array([gx, gy])

GOAL_KP = _goal_keypoints()

# ── Shared environment pool (one per thread) ──────────────────────────────────

_envs = {}

def _get_env():
    import threading
    tid = threading.get_ident()
    if tid not in _envs:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state")
        env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
        env.reset(seed=0)
        _envs[tid] = env
    return _envs[tid]


def _keypoints(block):
    pts = []
    for shape in block.shapes:
        for v in shape.get_vertices():
            w = v.rotated(shape.body.angle) + shape.body.position
            pts.append(np.array(w, dtype=np.float64))
    return np.vstack(pts)


def _macro_targets(raw_env):
    kp = _keypoints(raw_env.block)
    face_pts = np.array([
        (kp[0] + kp[1]) / 2, (kp[1] + kp[2]) / 2,
        (kp[5] + kp[6]) / 2, (kp[3] + kp[0]) / 2,
        kp[0], kp[1], kp[5], kp[6],
    ])
    cog = kp.mean(axis=0)
    d = face_pts - cog
    outward = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-6)
    approach = np.clip(face_pts + APPROACH_DIST * outward, WORKSPACE_LO, WORKSPACE_HI)
    push_end = np.clip(face_pts - PUSH_DEPTH * outward, WORKSPACE_LO, WORKSPACE_HI)
    return approach, push_end


def _restore_state(env, board):
    raw = env.unwrapped
    raw.agent.position = [float(board[0]), float(board[1])]
    raw.agent.velocity = (0.0, 0.0)
    raw.block.angle = float(board[4])
    raw.block.position = [float(board[2]), float(board[3])]
    raw.block.velocity = (0.0, 0.0)
    raw.block.angular_velocity = 0.0


def _read_state(env):
    raw = env.unwrapped
    return np.array(raw.get_obs(), dtype=np.float64)


# ── fast_step (Python-callable, not njit) ─────────────────────────────────────

def _keypoint_dist_to_goal(board):
    """Compute mean keypoint distance from board state (no pymunk needed)."""
    bx, by, angle = board[2], board[3], board[4]
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    cur_kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
    return float(np.mean(np.linalg.norm(cur_kp - GOAL_KP, axis=1)))


def _pusht_fast_step(board, action, player):
    """Execute one macro-action. Modifies board in-place. Returns (reward, winner, done, board)."""
    action = int(action)

    # No-op: return current IoU without moving
    if action == NUM_PUSH_DIRS:
        env = _get_env()
        _restore_state(env, board)
        cov = env.unwrapped._get_coverage()
        return float(cov), 0, False, board

    env = _get_env()
    _restore_state(env, board)

    # Adaptive push depth: shorter pushes when block is near goal
    dist = _keypoint_dist_to_goal(board)
    if dist < NEAR_DIST_THRESHOLD:
        scale = NEAR_PUSH_SCALE
    else:
        scale = 1.0
    n_approach = max(2, int(N_APPROACH * scale))
    n_push = max(2, int(N_PUSH * scale))
    n_settle = max(2, int(N_SETTLE * scale))

    raw = env.unwrapped
    approach, push_end = _macro_targets(raw)
    cov = 0.0
    for target, n in [(approach[action], n_approach),
                      (push_end[action], n_push),
                      (push_end[action], n_settle)]:
        for _ in range(n):
            _, _, _, _, info = env.step(target.astype(np.float32))
            cov = info.get("coverage", 0.0)

    new_state = _read_state(env)
    board[:] = new_state

    return float(cov), 0, False, board


def _pusht_valid_mask(board, player):
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    return mask


# ── GameLogic ─────────────────────────────────────────────────────────────────

class PushTLogic:
    NUM_ACTIONS     = NUM_ACTIONS       # 0-7 push directions + 8 no-op
    BOARD_SHAPE     = (5,)              # [agent_x, agent_y, block_x, block_y, block_angle]
    BOARD_DTYPE     = np.float64
    MAX_MOVES       = 20
    MAX_LEGAL_MOVES = NUM_ACTIONS
    PLAYER_1        = 1
    PLAYER_2        = 1                 # same as P1 (single-player)

    fast_step      = staticmethod(_pusht_fast_step)
    get_valid_mask = staticmethod(_pusht_valid_mask)

    def __init__(self, seed=42):
        self.seed = seed
        self._initial_board = None

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        env = _get_env()
        env.reset(seed=self.seed)
        self._initial_board = _read_state(env)

    def get_initial_board(self):
        if self._initial_board is None:
            self.reset()
        return self._initial_board.copy()


# ── MCTSModel ─────────────────────────────────────────────────────────────────

class PushTModel:
    """Uniform prior + simulator IoU as value."""

    def __init__(self, logic, diffusion_policy=None):
        self.logic = logic
        self.policy = diffusion_policy

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float()

        policy_out = torch.zeros(B, self.logic.NUM_ACTIONS)
        value_out = torch.zeros(B, 1)

        env = _get_env()
        for b in range(B):
            policy_out[b, :NUM_ACTIONS] = 1.0 / NUM_ACTIONS
            # Restore state and get actual IoU from the simulator
            board_np = boards[b].numpy().astype(np.float64)
            _restore_state(env, board_np)
            value_out[b] = env.unwrapped._get_coverage()

        return {"policy": policy_out, "value": value_out}


# ── Python-kernel PUCT / GumbelDense subclasses ───────────────────────────────

from gumbel_mcts.puct import PUCT, PUCTStorage
from gumbel_mcts.gumbel_dense import GumbelDense


class PythonPUCT(PUCT):
    """PUCT using pure-Python kernels (for environments with Python-only fast_step)."""

    def run_simulation_batch(self, model, active_games, num_simulations=50,
                             c_puct_base=19652, c_puct_init=1.25):
        from kernels.python_kernels import select_leaves_batch, backpropagate_batch
        game_indices = np.array(active_games, dtype=np.int32)
        logic = self.logic

        # Pre-expand roots
        unexpanded = [self.storage.root_indices[g]
                      for g in active_games
                      if not self.storage.is_expanded[self.storage.root_indices[g]]]
        if unexpanded:
            root_indices = np.array(unexpanded, dtype=np.int32)
            obs_boards = torch.tensor(self.storage.boards[root_indices],
                                      device=self.device, dtype=torch.float32)
            obs_players = torch.tensor(self.storage.players[root_indices],
                                       device=self.device, dtype=torch.long)
            batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
            with torch.no_grad():
                outputs = model.forward_for_mcts(batch)
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[root_indices] = True
            self.storage.prior_probs[root_indices] = priors
            backpropagate_batch(root_indices, vals,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)

        while True:
            min_visits = min(
                self.storage.visit_counts[self.storage.root_indices[g]]
                for g in active_games
            )
            if min_visits >= num_simulations + 1:
                break

            leaf_indices = select_leaves_batch(
                fast_step_func=logic.fast_step,
                get_valid_mask_func=logic.get_valid_mask,
                NUM_ACTIONS=logic.NUM_ACTIONS,
                player1=logic.PLAYER_1, player2=logic.PLAYER_2,
                game_indices=game_indices,
                root_indices=self.storage.root_indices,
                children=self.storage.children,
                visit_counts=self.storage.visit_counts,
                values=self.storage.values,
                prior_probs=self.storage.prior_probs,
                is_expanded=self.storage.is_expanded,
                is_terminal=self.storage.is_terminal,
                terminal_values=self.storage.terminal_values,
                boards=self.storage.boards,
                players=self.storage.players,
                parents=self.storage.parents,
                edge_from_parent=self.storage.edge_from_parent,
                next_free_idx_ptr=self.next_free_idx_arr,
                c_puct_base=c_puct_base, c_puct_init=c_puct_init,
                max_nodes=self.max_nodes,
                depths=self.storage.depths,
                max_game_depth=logic.MAX_MOVES
            )

            # Evaluate leaves
            is_term = self.storage.is_terminal[leaf_indices]
            leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
            if np.any(is_term):
                leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
            non_term = ~is_term
            if np.any(non_term):
                nn_idx = leaf_indices[non_term]
                obs_b = torch.tensor(self.storage.boards[nn_idx],
                                     device=self.device, dtype=torch.float32)
                obs_p = torch.tensor(self.storage.players[nn_idx],
                                     device=self.device, dtype=torch.long)
                batch = {"boards": obs_b.flatten(1), "current_player": obs_p}
                with torch.no_grad():
                    out = model.forward_for_mcts(batch)
                priors = out['policy'].float().cpu().numpy().astype(np.float64)
                vals = out['value'].float().cpu().numpy().flatten().astype(np.float64)
                self.storage.is_expanded[nn_idx] = True
                self.storage.prior_probs[nn_idx] = priors
                leaf_values[non_term] = vals

            backpropagate_batch(leaf_indices, leaf_values,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)


class PythonGumbelDense(GumbelDense):
    """GumbelDense using pure-Python single-player kernels."""

    def _expand_roots_v4(self, model, active_games):
        from kernels.python_kernels import backpropagate_batch
        n_active = len(active_games)
        root_indices = self.storage.root_indices[active_games]

        obs_boards = torch.tensor(self.storage.boards[root_indices],
                                  device=self.device, dtype=torch.float32)
        obs_players = torch.tensor(self.storage.players[root_indices],
                                   device=self.device, dtype=torch.long)
        batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}

        with torch.no_grad():
            outputs = model.forward_for_mcts(batch)

        probs = outputs['policy'].float().cpu().numpy()
        vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)

        self.storage.prior_probs[root_indices] = probs
        self.root_logits[:n_active] = np.log(probs + 1e-10)
        self.root_nn_values[:n_active] = vals.astype(np.float32)

        logic = model.logic
        for i in range(n_active):
            r_idx = root_indices[i]
            self.root_legal_masks[i] = logic.get_valid_mask(
                self.storage.boards[r_idx], self.storage.players[r_idx]
            )

        self.storage.is_expanded[root_indices] = True
        backpropagate_batch(root_indices, vals,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def _evaluate_and_backprop_v3(self, model, leaf_indices):
        from kernels.python_kernels import backpropagate_batch
        is_term = self.storage.is_terminal[leaf_indices]
        leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)

        if np.any(is_term):
            leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]

        non_term = ~is_term
        if np.any(non_term):
            nn_idx = leaf_indices[non_term]
            boards = torch.from_numpy(self.storage.boards[nn_idx]).to(self.device).float()
            players = torch.from_numpy(self.storage.players[nn_idx]).to(self.device).long()
            with torch.no_grad():
                outputs = model.forward_for_mcts(
                    {"boards": boards.flatten(1), "current_player": players})
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[nn_idx] = True
            self.storage.prior_probs[nn_idx] = priors
            leaf_values[non_term] = vals

        backpropagate_batch(leaf_indices, leaf_values,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def run_simulation_batch(self, model, active_games, num_simulations=50):
        from kernels.python_kernels import (
            descend_tree_kernel, backpropagate_batch,
            get_forced_root_moves_kernel, get_gumbel_score_kernel,
        )
        game_indices = np.array(active_games, dtype=np.int32)
        logic = model.logic
        n_active = len(active_games)

        self._expand_roots_v4(model, active_games)

        # Entropy-based dynamic max_k
        root_idxs = self.storage.root_indices[game_indices]
        raw_probs = self.storage.prior_probs[root_idxs].astype(np.float64)
        legal_f = self.root_legal_masks[:n_active].astype(np.float64)
        legal_probs = raw_probs * legal_f
        legal_probs /= legal_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        mean_ent = float(-(legal_probs * np.log(legal_probs + 1e-10)).sum(axis=1).mean())
        mean_n_legal = float(legal_f.sum(axis=1).mean())
        max_ent = np.log(mean_n_legal) if mean_n_legal > 1.0 else 1.0
        ratio = min(1.0, mean_ent / max(float(max_ent), 1e-8))
        dynamic_max_k = int(round(
            self.max_considered_actions
            - ratio * (self.max_considered_actions - self.entropy_min_k)
        ))
        dynamic_max_k = max(self.entropy_min_k, min(self.max_considered_actions, dynamic_max_k))
        self.last_root_entropy = mean_ent
        self.last_dynamic_max_k = dynamic_max_k

        max_k = min(self.storage.num_actions, dynamic_max_k)
        num_phases = max(1, int(np.log2(max_k)))
        first_phase_budget = num_simulations // num_phases
        k_initial = min(max_k, first_phase_budget // 2)
        k_initial = max(2, k_initial)
        num_phases = max(1, int(np.log2(k_initial)))
        k_initial = min(k_initial, max(2, (num_simulations // num_phases) // 4))
        k_initial = max(2, 1 << int(np.log2(k_initial)))
        num_phases = max(1, int(np.log2(k_initial)))

        candidate_mask = self._get_initial_gumbel_candidates(logic, game_indices, k_initial)

        remaining = num_simulations
        for phase in range(num_phases):
            k_phase = max(1, k_initial // (2 ** phase))
            phases_left = num_phases - phase
            budget_this_phase = remaining if phase == num_phases - 1 else remaining // phases_left
            sims_per_action = max(1, budget_this_phase // k_phase)

            for candidate_rank in range(k_phase):
                root_moves = get_forced_root_moves_kernel(
                    n_active, candidate_mask, candidate_rank
                )
                for _ in range(sims_per_action):
                    leaf_indices = descend_tree_kernel(
                        logic.fast_step, logic.get_valid_mask,
                        self.storage.num_actions,
                        logic.PLAYER_1, logic.PLAYER_2,
                        game_indices,
                        self.storage.root_indices[game_indices],
                        root_moves,
                        self.storage.children, self.storage.visit_counts,
                        self.storage.values, self.storage.prior_probs,
                        self.storage.is_expanded, self.storage.is_terminal,
                        self.storage.terminal_values, self.storage.boards,
                        self.storage.players, self.storage.parents,
                        self.storage.edge_from_parent, self.next_free_idx_arr,
                        self.max_nodes, self.storage.depths, logic.MAX_MOVES,
                        self.c_visit, self.c_scale
                    )
                    self._evaluate_and_backprop_v3(model, leaf_indices)

            remaining -= (k_phase * sims_per_action)

            if phase < num_phases - 1:
                candidate_mask = self._halve_candidates_py(
                    game_indices, candidate_mask, get_gumbel_score_kernel
                )

        final_moves = self._get_final_survivors_py(
            game_indices, candidate_mask, get_gumbel_score_kernel
        )
        return final_moves

    def _halve_candidates_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values,
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        new_mask = np.zeros_like(candidate_mask)
        for i in range(n_active):
            active_moves = np.where(candidate_mask[i])[0]
            if len(active_moves) <= 1:
                new_mask[i] = candidate_mask[i]
                continue
            num_to_keep = max(1, len(active_moves) // 2)
            row_scores = scores[i, active_moves]
            top_indices = np.argsort(row_scores)[-num_to_keep:]
            new_mask[i, active_moves[top_indices]] = True
        return new_mask

    def _get_final_survivors_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values,
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        return np.array([np.argmax(scores[i]) for i in range(n_active)], dtype=np.int32)
