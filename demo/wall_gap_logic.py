"""
Wall-gap PushT wrapped as a GameLogic for the gumbel_mcts library.

Micro-action design matching pusht_logic.py:
7 touch points × 3 push directions + no-op = 22 actions.
Reposition-then-push: agent teleports to approach point before pushing.
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
    NUM_PUSH_DIRS, NUM_ACTIONS, BASE_STEP_SIZE, APPROACH_DIST, N_PHYSICS,
    WORKSPACE_LO, WORKSPACE_HI,
    _keypoints, _compute_push_targets, PythonPUCT, PythonGumbelDense,
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


# ── fast_step with wall enforcement + reposition-then-push ───────────────────

def _wall_gap_fast_step(board, action, player, gap_size=200):
    action = int(action)
    env = _get_wall_env(gap_size)
    _restore_state(env, board)
    raw = env.unwrapped

    frac_before = _fraction_past_wall(board)
    board_before = board.copy()

    if action < NUM_PUSH_DIRS:
        approaches, push_targets = _compute_push_targets(raw)

        # Reposition agent to approach point
        raw.agent.position = list(approaches[action])
        raw.agent.velocity = (0, 0)

        target = push_targets[action].astype(np.float32)
        agent_x_limit = WALL_X - 15 - 15
        half = gap_size / 2
        wall_left = WALL_X - 15

        for _ in range(N_PHYSICS):
            prev_pos = list(raw.block.position)
            prev_angle = raw.block.angle

            env.step(target)

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
    """Geometric prior (push toward wall gap) + fraction_past_wall as value."""

    def __init__(self, logic, use_geometric_prior=True):
        self.logic = logic
        self.use_geometric_prior = use_geometric_prior

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float().cpu()

        policy_out = torch.zeros(B, self.logic.NUM_ACTIONS)
        value_out = torch.zeros(B, 1)

        for b in range(B):
            board_np = boards[b].numpy().astype(np.float64)
            frac = _fraction_past_wall(board_np)
            value_out[b] = frac

            if self.use_geometric_prior:
                env = _get_wall_env(self.logic.gap_size)
                _restore_state(env, board_np)
                raw = env.unwrapped

                # Goal direction: push block toward the gap center
                block_pos = np.array([board_np[2], board_np[3]])
                # Target: gap center, on the far side of the wall
                gap_target = np.array([WALL_X + 50, GAP_CENTER_Y])
                to_goal = gap_target - block_pos
                to_goal_norm = np.linalg.norm(to_goal)
                if to_goal_norm > 1e-6:
                    to_goal /= to_goal_norm

                approaches, push_targets = _compute_push_targets(raw)
                scores = np.zeros(NUM_PUSH_DIRS, dtype=np.float32)
                for a in range(NUM_PUSH_DIRS):
                    push_dir = push_targets[a] - approaches[a]
                    pdn = np.linalg.norm(push_dir)
                    if pdn > 1e-6:
                        push_dir /= pdn
                    scores[a] = np.dot(push_dir, to_goal)
                scores -= scores.max()
                prior = np.zeros(NUM_ACTIONS, dtype=np.float32)
                prior[:NUM_PUSH_DIRS] = np.exp(scores * 1.0)
                prior[NUM_PUSH_DIRS] = np.mean(prior[:NUM_PUSH_DIRS])
                prior /= prior.sum()
                policy_out[b] = torch.from_numpy(prior)
            else:
                policy_out[b] = 1.0 / NUM_ACTIONS

        return {"policy": policy_out, "value": value_out}
