"""
Wall-gap PushT wrapped as a GameLogic for the gumbel_mcts library.

Reuses PythonPUCT / PythonGumbelDense from pusht_logic.py.
The wall enforcement (per-substep block rollback + agent clipping) is
baked into fast_step so the tree search respects the constraint.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import pymunk
import gymnasium as gym
import gym_pusht  # noqa: F401

from pusht_logic import (
    NUM_PUSH_DIRS, NUM_ACTIONS, APPROACH_DIST, PUSH_DEPTH,
    N_APPROACH, N_PUSH, N_SETTLE, WORKSPACE_LO, WORKSPACE_HI,
    LOCAL_VERTS, _keypoints, PythonPUCT, PythonGumbelDense,
)

# ── Wall constants ───────────────────────────────────────────────────────────

WALL_X       = 300.0
GAP_CENTER_Y = 256.0

# ── Per-gap-size environment pool ────────────────────────────────────────────

_wall_envs = {}

def _get_wall_env(gap_size):
    import threading
    key = (threading.get_ident(), gap_size)
    if key not in _wall_envs:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state")
        env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
        env.reset(seed=0)
        _wall_envs[key] = env
    return _wall_envs[key]


def _setup_walls(env, gap_size):
    """Add wall bodies to the pymunk space. Idempotent per env."""
    raw = env.unwrapped
    space = raw.block._space
    half = gap_size / 2
    wall_width = 30

    # Remove old wall bodies if any
    for body in list(space.bodies):
        if getattr(body, '_is_wall', False):
            for shape in body.shapes:
                space.remove(shape)
            space.remove(body)

    for center_y, h in [
        (GAP_CENTER_Y + half + (512 - GAP_CENTER_Y - half) / 2,
         512 - GAP_CENTER_Y - half),
        ((GAP_CENTER_Y - half) / 2, GAP_CENTER_Y - half),
    ]:
        if h > 0:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (WALL_X, center_y)
            body._is_wall = True
            box = pymunk.Poly.create_box(body, size=(wall_width, h))
            box.friction = 1.0
            box.elasticity = 0.0
            space.add(body, box)


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


def _fraction_past_wall(board):
    bx, by, angle = board[2], board[3], board[4]
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
    return float(np.mean(kp[:, 0] > WALL_X))


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


# ── fast_step with wall enforcement ──────────────────────────────────────────

def _wall_gap_fast_step(board, action, player, gap_size=200):
    action = int(action)

    env = _get_wall_env(gap_size)
    _restore_state(env, board)
    raw = env.unwrapped

    if action == NUM_PUSH_DIRS:
        frac = _fraction_past_wall(board)
        return float(frac), 0, False, board

    agent_x_limit = WALL_X - 15 - 15
    half = gap_size / 2
    wall_left = WALL_X - 15

    approach, push_end = _macro_targets(raw)
    for target, n in [(approach[action], N_APPROACH),
                      (push_end[action], N_PUSH),
                      (push_end[action], N_SETTLE)]:
        clipped = target.copy()
        clipped[0] = min(clipped[0], agent_x_limit)
        for _ in range(n):
            prev_pos = list(raw.block.position)
            prev_angle = raw.block.angle

            env.step(clipped.astype(np.float32))

            # Check block-wall overlap
            bx, by = raw.block.position
            ba = raw.block.angle
            c, s = np.cos(ba), np.sin(ba)
            R = np.array([[c, -s], [s, c]])
            kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
            violation = False
            for kx, ky in kp:
                if kx > wall_left and (ky > GAP_CENTER_Y + half or ky < GAP_CENTER_Y - half):
                    violation = True
                    break
            if violation:
                raw.block.angle = prev_angle
                raw.block.position = prev_pos
                raw.block.velocity = (0, 0)
                raw.block.angular_velocity = 0

            # Clamp agent
            ax, ay = raw.agent.position
            if ax > agent_x_limit:
                in_gap = (GAP_CENTER_Y - half) < ay < (GAP_CENTER_Y + half)
                if not in_gap:
                    raw.agent.position = (agent_x_limit, ay)

    new_state = _read_state(env)
    board[:] = new_state
    frac = _fraction_past_wall(board)
    return float(frac), 0, False, board


def _wall_gap_valid_mask(board, player):
    return np.ones(NUM_ACTIONS, dtype=np.float32)


# ── GameLogic ────────────────────────────────────────────────────────────────

class WallGapLogic:
    NUM_ACTIONS     = NUM_ACTIONS
    BOARD_SHAPE     = (5,)
    BOARD_DTYPE     = np.float64
    MAX_MOVES       = 20
    MAX_LEGAL_MOVES = NUM_ACTIONS
    PLAYER_1        = 1
    PLAYER_2        = 1

    get_valid_mask = staticmethod(_wall_gap_valid_mask)

    def __init__(self, gap_size=200, seed=42):
        self.gap_size = gap_size
        self.seed = seed
        self._initial_board = None
        self.fast_step = lambda board, action, player: _wall_gap_fast_step(
            board, action, player, gap_size=self.gap_size
        )

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        env = _get_wall_env(self.gap_size)
        env.reset(seed=self.seed)
        raw = env.unwrapped
        raw.goal_pose = np.array([-1000.0, -1000.0, 0.0])
        raw.block.angle = 0.0
        raw.block.position = (150, 256)
        raw.block.velocity = (0, 0)
        raw.block.angular_velocity = 0
        raw.agent.position = (80, 256)
        _setup_walls(env, self.gap_size)
        raw.block._space.step(0.001)
        raw.block.velocity = (0, 0)
        raw.block.angular_velocity = 0
        self._initial_board = _read_state(env)

    def get_initial_board(self):
        if self._initial_board is None:
            self.reset()
        return self._initial_board.copy()


# ── MCTSModel ────────────────────────────────────────────────────────────────

class WallGapModel:
    """Diffusion prior (or uniform) + fraction_past_wall as value."""

    def __init__(self, logic, diffusion_policy=None):
        self.logic = logic
        self.policy = diffusion_policy
        self._obs_history = []

    def reset_obs_history(self):
        self._obs_history = []

    def _compute_prior_from_board(self, board_np):
        from pusht import _compute_prior, _keypoints as kp_fn
        env = _get_wall_env(self.logic.gap_size)
        _restore_state(env, board_np)
        raw = env.unwrapped
        obs = {
            "environment_state": kp_fn(raw.block).flatten().astype(np.float32),
            "agent_pos": np.array(raw.agent.position, dtype=np.float32),
        }
        if not self._obs_history:
            self._obs_history.append(obs)
        prior_8 = _compute_prior(self.policy, raw, self._obs_history)
        prior = np.ones(NUM_ACTIONS, dtype=np.float32)
        prior[:NUM_PUSH_DIRS] = prior_8
        prior[NUM_PUSH_DIRS] = 0.02
        prior /= prior.sum()
        return prior

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float()

        policy_out = torch.zeros(B, self.logic.NUM_ACTIONS)
        value_out = torch.zeros(B, 1)

        for b in range(B):
            board_np = boards[b].numpy().astype(np.float64)
            value_out[b] = _fraction_past_wall(board_np)
            if self.policy is not None:
                prior = self._compute_prior_from_board(board_np)
                policy_out[b] = torch.from_numpy(prior)
            else:
                policy_out[b] = 1.0 / NUM_ACTIONS

        return {"policy": policy_out, "value": value_out}
