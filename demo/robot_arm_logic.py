"""
2D robotic arm with 3 joints — GameLogic for gumbel_mcts library.

The arm has a fixed base, 3 links, and must reach a target point.
Joint angles are capped to [-90°, +90°]. Self-collision is penalized.

Action space: 3 joints × 2 directions (±step) + no-op = 7 actions.
Board state: [j0, j1, j2, target_x, target_y] (angles in radians).
"""

import numpy as np
import torch
from PIL import Image, ImageDraw

# ── Constants ────────────────────────────────────────────────────────────────

LINK_LENGTHS = np.array([100.0, 80.0, 60.0])
N_JOINTS     = 3
ANGLE_STEP   = np.radians(10)  # 10° per action
ANGLE_LIMIT  = np.radians(90)  # ±90° per joint
BASE_POS     = np.array([256.0, 256.0])
WORKSPACE    = 512

NUM_ACTIONS  = N_JOINTS * 2 + 1  # 7: j0+, j0-, j1+, j1-, j2+, j2-, noop
SUCCESS_DIST = 15.0  # pixels

# Obstacles: list of (center_x, center_y, radius)
OBSTACLES = [
    (310, 210, 20),
]

# ── Forward kinematics ──────────────────────────────────────────────────────

def forward_kinematics(angles):
    """Compute joint positions from angles. Returns (4, 2) array: base + 3 joints."""
    positions = np.zeros((N_JOINTS + 1, 2))
    positions[0] = BASE_POS
    cumulative_angle = 0.0
    for i in range(N_JOINTS):
        cumulative_angle += angles[i]
        dx = LINK_LENGTHS[i] * np.cos(cumulative_angle)
        dy = LINK_LENGTHS[i] * np.sin(cumulative_angle)
        positions[i + 1] = positions[i] + np.array([dx, dy])
    return positions


def end_effector(angles):
    """Get end-effector position."""
    return forward_kinematics(angles)[-1]


def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    # Exclude endpoints (adjacent links share a joint)
    return 0.01 < t < 0.99 and 0.01 < u < 0.99


def segment_circle_intersect(p1, p2, center, radius):
    """Check if segment p1-p2 intersects circle (center, radius)."""
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def has_collision(angles):
    """Check self-collision and obstacle collision."""
    positions = forward_kinematics(angles)
    # Self-collision
    for i in range(N_JOINTS):
        for j in range(i + 2, N_JOINTS):
            if segments_intersect(positions[i], positions[i+1],
                                  positions[j], positions[j+1]):
                return True
    # Obstacle collision
    for i in range(N_JOINTS):
        for cx, cy, r in OBSTACLES:
            if segment_circle_intersect(positions[i], positions[i+1],
                                        np.array([cx, cy]), r):
                return True
    return False


# ── Rendering ────────────────────────────────────────────────────────────────

def render(angles, target):
    """Render the arm and target as a PIL Image."""
    img = Image.new("RGB", (WORKSPACE, WORKSPACE), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid
    for i in range(0, WORKSPACE, 64):
        draw.line([(i, 0), (i, WORKSPACE)], fill=(230, 230, 230), width=1)
        draw.line([(0, i), (WORKSPACE, i)], fill=(230, 230, 230), width=1)

    # Obstacles
    for cx, cy, r in OBSTACLES:
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(180, 180, 180), outline=(120, 120, 120), width=2)

    # Target
    tx, ty = int(target[0]), int(target[1])
    draw.ellipse([tx-8, ty-8, tx+8, ty+8], fill=(80, 200, 80), outline=(40, 150, 40), width=2)

    # Arm
    positions = forward_kinematics(angles)
    link_colors = [(70, 130, 200), (50, 110, 180), (30, 90, 160)]
    for i in range(N_JOINTS):
        x1, y1 = int(positions[i][0]), int(positions[i][1])
        x2, y2 = int(positions[i+1][0]), int(positions[i+1][1])
        draw.line([(x1, y1), (x2, y2)], fill=link_colors[i], width=6)

    # Joints
    for i in range(N_JOINTS + 1):
        x, y = int(positions[i][0]), int(positions[i][1])
        r = 5 if i > 0 else 7
        color = (200, 60, 60) if i == 0 else (50, 50, 50)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(30, 30, 30), width=1)

    # End effector highlight
    ex, ey = int(positions[-1][0]), int(positions[-1][1])
    draw.ellipse([ex-6, ey-6, ex+6, ey+6], fill=(255, 100, 50), outline=(200, 60, 30), width=2)

    return np.array(img)


# ── GameLogic ────────────────────────────────────────────────────────────────

def _arm_fast_step(board, action, player):
    """Execute one action. Board = [j0, j1, j2, target_x, target_y]."""
    action = int(action)
    angles = board[:N_JOINTS].copy()
    target = board[N_JOINTS:N_JOINTS+2]

    if action < N_JOINTS * 2:
        joint = action // 2
        direction = 1 if action % 2 == 0 else -1
        new_angle = angles[joint] + direction * ANGLE_STEP
        new_angle = np.clip(new_angle, -ANGLE_LIMIT, ANGLE_LIMIT)
        angles[joint] = new_angle

    # Check collisions (self + obstacles)
    if has_collision(angles):
        return 0.0, 0, True, board  # terminal, bad

    board[:N_JOINTS] = angles

    # Distance to target
    ee = end_effector(angles)
    dist = np.linalg.norm(ee - target)
    value = max(0.0, 1.0 - dist / 300.0)

    done = dist < SUCCESS_DIST
    return float(value), int(done), done, board


def _arm_valid_mask(board, player):
    return np.ones(NUM_ACTIONS, dtype=np.float32)


class RobotArmLogic:
    NUM_ACTIONS     = NUM_ACTIONS
    BOARD_SHAPE     = (5,)  # [j0, j1, j2, target_x, target_y]
    BOARD_DTYPE     = np.float64
    MAX_MOVES       = 100
    MAX_LEGAL_MOVES = NUM_ACTIONS
    PLAYER_1        = 1
    PLAYER_2        = 1

    fast_step      = staticmethod(_arm_fast_step)
    get_valid_mask = staticmethod(_arm_valid_mask)

    def __init__(self, seed=42):
        self.seed = seed
        self._initial_board = None

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        # Random target within reachable workspace
        max_reach = LINK_LENGTHS.sum()
        while True:
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(50, max_reach * 0.9)
            target = BASE_POS + np.array([np.cos(angle), np.sin(angle)]) * dist
            if not (20 < target[0] < WORKSPACE - 20 and 20 < target[1] < WORKSPACE - 20):
                continue
            # Don't place target inside an obstacle
            in_obs = False
            for cx, cy, r in OBSTACLES:
                if np.sqrt((target[0]-cx)**2 + (target[1]-cy)**2) < r + 20:
                    in_obs = True; break
            if not in_obs:
                break
        self._initial_board = np.zeros(5, dtype=np.float64)
        self._initial_board[N_JOINTS:N_JOINTS+2] = target

    def get_initial_board(self):
        if self._initial_board is None:
            self.reset()
        return self._initial_board.copy()


# ── MCTSModel ────────────────────────────────────────────────────────────────

class RobotArmModel:
    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float().cpu()

        policy_out = torch.zeros(B, NUM_ACTIONS)
        value_out = torch.zeros(B, 1)

        for b in range(B):
            board_np = boards[b].numpy().astype(np.float64)
            angles = board_np[:N_JOINTS]
            target = board_np[N_JOINTS:N_JOINTS+2]

            ee = end_effector(angles)
            dist = np.linalg.norm(ee - target)
            value_out[b] = max(0.0, 1.0 - dist / 300.0)
            policy_out[b] = 1.0 / NUM_ACTIONS

        return {"policy": policy_out, "value": value_out}
