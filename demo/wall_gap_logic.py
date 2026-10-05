"""
Wall-gap PushT wrapped as a GameLogic for the gumbel_mcts library.

Micro-action design matching pusht_logic.py:
7 touch points × 3 push directions + no-op = 22 actions.
Constant step_size=10px, N_PHYSICS=10.
Wall enforcement via per-substep block rollback + agent clipping.
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
    LOCAL_VERTS, LOCAL_TOUCH_POINTS, N_TOUCH, N_DIRS,
    NUM_PUSH_DIRS, NUM_ACTIONS, BASE_STEP_SIZE, N_PHYSICS,
    WORKSPACE_LO, WORKSPACE_HI,
    _keypoints, PythonPUCT, PythonGumbelDense,
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
    raw = env.unwrapped
    space = raw.block._space
    half = gap_size / 2
    wall_width = 30

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
    return np.array(env.unwrapped.get_obs(), dtype=np.float64)


def _fraction_past_wall(board):
    bx, by, angle = board[2], board[3], board[4]
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
    return float(np.mean(kp[:, 0] > WALL_X))


def _block_moved(board_before, board_after, threshold=0.5):
    dx = abs(board_after[2] - board_before[2])
    dy = abs(board_after[3] - board_before[3])
    da = abs(board_after[4] - board_before[4])
    return (dx + dy) > threshold or da > 0.01


# ── Compute push targets (micro-actions) ─────────────────────────────────────

def _compute_push_targets(raw_env):
    step_size = BASE_STEP_SIZE
    block = raw_env.block
    angle = block.angle
    bx, by = block.position
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    world_pts = (R @ LOCAL_TOUCH_POINTS.T).T + np.array([bx, by])
    cog = (R @ LOCAL_VERTS.T).T.mean(axis=0) + np.array([bx, by])

    targets = []
    for pt in world_pts:
        d = pt - cog
        outward = d / max(np.linalg.norm(d), 1e-6)
        tangent_cw = np.array([outward[1], -outward[0]])
        tangent_ccw = np.array([-outward[1], outward[0]])
        for direction in [outward, tangent_cw, tangent_ccw]:
            agent_target = pt + direction * step_size
            targets.append(np.clip(agent_target, WORKSPACE_LO, WORKSPACE_HI))

    return np.array(targets)


# ── fast_step with wall enforcement ──────────────────────────────────────────

def _wall_gap_fast_step(board, action, player, gap_size=200):
    action = int(action)
    env = _get_wall_env(gap_size)
    _restore_state(env, board)
    raw = env.unwrapped

    frac_before = _fraction_past_wall(board)
    board_before = board.copy()

    if action < NUM_PUSH_DIRS:
        targets = _compute_push_targets(raw)
        agent_target = targets[action].astype(np.float32)

        agent_x_limit = WALL_X - 15 - 15
        half = gap_size / 2
        wall_left = WALL_X - 15

        for _ in range(N_PHYSICS):
            prev_pos = list(raw.block.position)
            prev_angle = raw.block.angle

            env.step(agent_target)

            # Block boundary clamp
            bx, by = raw.block.position
            raw.block.position = (max(60, min(452, bx)), max(60, min(452, by)))

            # Wall violation check
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

            # Agent wall clamp
            ax, ay = raw.agent.position
            if ax > agent_x_limit:
                in_gap = (GAP_CENTER_Y - half) < ay < (GAP_CENTER_Y + half)
                if not in_gap:
                    raw.agent.position = (agent_x_limit, ay)

    new_state = _read_state(env)
    board[:] = new_state
    frac_after = _fraction_past_wall(board)

    if not _block_moved(board_before, board):
        if frac_before >= 0.1:
            return float(frac_before), 0, False, board
        return 0.0, 0, True, board

    return float(frac_after), 0, False, board


def _wall_gap_valid_mask(board, player):
    return np.ones(NUM_ACTIONS, dtype=np.float32)


# ── GameLogic ────────────────────────────────────────────────────────────────

class WallGapLogic:
    NUM_ACTIONS     = NUM_ACTIONS
    BOARD_SHAPE     = (5,)
    BOARD_DTYPE     = np.float64
    MAX_MOVES       = 200
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
    """Uniform prior + fraction_past_wall as value."""

    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float().cpu()

        policy_out = torch.zeros(B, self.logic.NUM_ACTIONS)
        value_out = torch.zeros(B, 1)

        for b in range(B):
            board_np = boards[b].numpy().astype(np.float64)
            frac = _fraction_past_wall(board_np)
            value_out[b] = frac
            policy_out[b] = 1.0 / NUM_ACTIONS

        return {"policy": policy_out, "value": value_out}
