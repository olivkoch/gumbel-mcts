"""
demo/pusht_wall.py — PushT wall-gap task: push the T-block through a hole in a wall.

Tests whether MCTS can identify safe root actions (those that don't crash the
block into the wall) before committing — the setting where Gumbel's sequential
halving should outperform PUCT's depth-first exploration.

Uses the same lerobot/diffusion_pusht_keypoints prior as the PushT demo.

Usage:
    uv run python demo/pusht_wall.py --gap-size 200 --budget 32
    uv run python demo/pusht_wall.py --gap-size 80 --budget 64
"""

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pymunk
import gymnasium as gym
import gym_pusht  # noqa: F401
from PIL import Image

# Reuse pusht.py's GIF maker and diffusion prior
sys.path.insert(0, os.path.dirname(__file__))
from pusht import make_gif, _load_policy, _compute_prior, _keypoints, _make_obs

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_PUSH_DIRS = 8
APPROACH_DIST = 80
PUSH_DEPTH    = 60
N_APPROACH    = 12
N_PUSH        = 18
N_SETTLE      = 12

LOCAL_VERTS = np.array([
    [-60, 0], [60, 0], [60, 30], [-60, 30],
    [-15, 30], [15, 30], [15, 120], [-15, 120],
], dtype=np.float64)

WALL_X       = 300.0
GAP_CENTER_Y = 256.0
WALL_THICK   = 5


# ── Environment ───────────────────────────────────────────────────────────────

class WallGapEnv:
    def __init__(self, gap_size=200, seed=42, record=False):
        self.gap_size = gap_size
        self.seed = seed
        self.record = record
        _env = gym.make("gym_pusht/PushT-v0", obs_type="state",
                        render_mode="rgb_array" if record else None)
        self.env = _env.env if isinstance(_env, gym.wrappers.TimeLimit) else _env
        self._rng = np.random.default_rng(seed)
        self.frames = []
        self._wall_bodies = []

    def _add_walls(self):
        space = self.env.unwrapped.block._space
        half = self.gap_size / 2
        wall_width = 30  # thick enough to prevent tunneling

        # Top wall: solid box from gap top edge to ceiling
        top_h = 512 - (GAP_CENTER_Y + half)
        if top_h > 0:
            top_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            top_body.position = (WALL_X, GAP_CENTER_Y + half + top_h / 2)
            top_box = pymunk.Poly.create_box(top_body, size=(wall_width, top_h))
            top_box.friction = 1.0
            top_box.elasticity = 0.0
            space.add(top_body, top_box)
            self._wall_bodies.append(top_body)

        # Bottom wall: solid box from floor to gap bottom edge
        bot_h = GAP_CENTER_Y - half
        if bot_h > 0:
            bot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            bot_body.position = (WALL_X, bot_h / 2)
            bot_box = pymunk.Poly.create_box(bot_body, size=(wall_width, bot_h))
            bot_box.friction = 1.0
            bot_box.elasticity = 0.0
            space.add(bot_body, bot_box)
            self._wall_bodies.append(bot_body)

    def reset(self):
        self.env.reset(seed=self.seed)
        raw = self.env.unwrapped
        # Hide the goal overlay (not used in wall-gap task)
        raw.goal_pose = np.array([-1000.0, -1000.0, 0.0])
        # Place block on the left side, bar facing the wall (needs rotation).
        # Angle must be set BEFORE position (COG offset issue in pymunk).
        raw.block.angle = 0.0   # bar horizontal, stem up — widest face toward wall
        raw.block.position = (150, 256)
        raw.block.velocity = (0, 0)
        raw.block.angular_velocity = 0
        raw.agent.position = (80, 256)
        self._wall_bodies = []
        self._add_walls()
        # Step physics once to sync pymunk state with renderer
        raw.block._space.step(0.001)
        raw.block.velocity = (0, 0)
        raw.block.angular_velocity = 0
        self.frames.clear()
        if self.record:
            self.frames.append(self.env.render())

    def close(self):
        self.env.close()

    def state(self):
        raw = self.env.unwrapped
        return np.array(raw.get_obs(), dtype=np.float64)

    def fraction_past_wall(self):
        raw = self.env.unwrapped
        bx, by, ba = raw.block.position[0], raw.block.position[1], raw.block.angle
        c, s = np.cos(ba), np.sin(ba)
        R = np.array([[c, -s], [s, c]])
        kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
        return float(np.mean(kp[:, 0] > WALL_X))

    def _keypoints(self):
        raw = self.env.unwrapped
        pts = []
        for shape in raw.block.shapes:
            for v in shape.get_vertices():
                w = v.rotated(shape.body.angle) + shape.body.position
                pts.append(np.array(w, dtype=np.float64))
        return np.vstack(pts)

    def _macro_targets(self):
        kp = self._keypoints()
        face_pts = np.array([
            (kp[0] + kp[1]) / 2, (kp[1] + kp[2]) / 2,
            (kp[5] + kp[6]) / 2, (kp[3] + kp[0]) / 2,
            kp[0], kp[1], kp[5], kp[6],
        ])
        cog = kp.mean(axis=0)
        d = face_pts - cog
        outward = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-6)
        approach = np.clip(face_pts + APPROACH_DIST * outward, 0, 512)
        push_end = np.clip(face_pts - PUSH_DEPTH * outward, 0, 512)
        return approach, push_end

    def step(self, action):
        """Execute a macro-action. Returns fraction_past_wall."""
        raw = self.env.unwrapped

        if action == NUM_PUSH_DIRS:  # no-op
            if self.record:
                self.frames.append(self.env.render())
            return self.fraction_past_wall()

        approach, push_end = self._macro_targets()
        agent_x_limit = WALL_X - 15 - 15
        half = self.gap_size / 2
        wall_left = WALL_X - 15
        step_count = 0
        for target, n in [(approach[action], N_APPROACH),
                          (push_end[action], N_PUSH),
                          (push_end[action], N_SETTLE)]:
            clipped = target.copy()
            clipped[0] = min(clipped[0], agent_x_limit)
            for si in range(n):
                # Save state before step
                prev_block_pos = list(raw.block.position)
                prev_block_angle = raw.block.angle
                prev_block_vel = list(raw.block.velocity)
                prev_block_angvel = raw.block.angular_velocity

                self.env.step(clipped.astype(np.float32))

                # Check if block overlaps wall (not gap)
                bx, by = raw.block.position
                ba = raw.block.angle
                c, s = np.cos(ba), np.sin(ba)
                R = np.array([[c, -s], [s, c]])
                kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
                wall_violation = False
                for kx, ky in kp:
                    if kx > wall_left and (ky > GAP_CENTER_Y + half or ky < GAP_CENTER_Y - half):
                        wall_violation = True
                        break

                if wall_violation:
                    raw.block.angle = prev_block_angle
                    raw.block.position = prev_block_pos
                    raw.block.velocity = (0, 0)
                    raw.block.angular_velocity = 0

                # Clamp agent
                ax, ay = raw.agent.position
                if ax > agent_x_limit:
                    in_gap = (GAP_CENTER_Y - half) < ay < (GAP_CENTER_Y + half)
                    if not in_gap:
                        raw.agent.position = (agent_x_limit, ay)

                step_count += 1
                if self.record and step_count % 3 == 0:
                    self.frames.append(self.env.render())

        return self.fraction_past_wall()


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, strategy, n_macros, budget, policy=None):
    """Run one episode with flat-bandit MCTS. Returns list of fraction_past_wall."""
    env.reset()
    raw = env.env.unwrapped
    obs_history = [_make_obs(raw)]
    results = []
    n_actions = NUM_PUSH_DIRS + 1

    for step_i in range(n_macros):
        root_state = env.state()
        prior = _compute_prior(policy, raw, obs_history)
        # Extend prior to include no-op action with low weight
        prior_full = np.ones(n_actions, dtype=np.float32)
        prior_full[:NUM_PUSH_DIRS] = prior
        prior_full[NUM_PUSH_DIRS] = 0.02  # small no-op weight
        prior_full /= prior_full.sum()

        t0 = time.perf_counter()

        if strategy == "random":
            action = int(env._rng.integers(n_actions))
        elif strategy == "puct":
            action = _puct_select(env, root_state, prior_full, budget)
        elif strategy == "gumbel":
            action = _gumbel_select_sh(env, root_state, prior_full, budget)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        _restore(env, root_state)
        frac = env.step(action)
        dt = time.perf_counter() - t0
        results.append(frac)
        obs_history.append(_make_obs(raw))

        prior_str = " ".join(f"{p:.2f}" for p in prior_full)
        print(f"[{strategy:6s}] step {step_i+1}/{n_macros} | action={action} | "
              f"frac={frac:.3f} | plan={dt:.2f}s")
        print(f"          prior=[{prior_str}]")

    return results


def _puct_select(env, root_state, prior, budget):
    """PUCT with Q updates and prior-weighted UCB."""
    n_actions = len(prior)
    N = np.zeros(n_actions, dtype=float)
    Q = np.zeros(n_actions, dtype=float)
    was_recording = env.record
    env.record = False
    for _ in range(budget):
        N_total = max(1.0, N.sum())
        ucb = Q + 1.5 * prior * np.sqrt(N_total) / (1.0 + N)
        a = int(np.argmax(ucb))
        _restore(env, root_state)
        frac = env.step(a)
        N[a] += 1.0
        Q[a] += (frac - Q[a]) / N[a]
    env.record = was_recording
    return int(np.argmax(N))


def _gumbel_select_sh(env, root_state, prior, budget):
    """Gumbel sequential halving with log-prior perturbation."""
    n_actions = len(prior)
    log_prior = np.log(prior.clip(min=1e-8))
    u = np.random.uniform(1e-8, 1 - 1e-8, n_actions)
    gumbel = log_prior - np.log(-np.log(u))

    k = min(8, n_actions)
    phases = max(1, int(np.log2(k)))

    candidates = list(np.argsort(gumbel)[::-1][:k])
    visits = np.zeros(n_actions, dtype=int)
    total_q = np.zeros(n_actions)
    remaining = budget

    was_recording = env.record
    env.record = False
    for phase in range(phases):
        k_phase = len(candidates)
        b = remaining if phase == phases - 1 else remaining // (phases - phase)
        spc = max(1, b // k_phase)
        for a in candidates:
            for _ in range(spc):
                if remaining <= 0:
                    break
                _restore(env, root_state)
                frac = env.step(a)
                visits[a] += 1
                total_q[a] += frac
                remaining -= 1
        if phase < phases - 1 and len(candidates) > 1:
            mean_q = np.array([total_q[a] / visits[a] if visits[a] > 0 else 0.0
                               for a in candidates])
            g = np.array([gumbel[a] for a in candidates])
            scores = mean_q + g
            half = max(1, len(candidates) // 2)
            top = np.argsort(scores)[::-1][:half]
            candidates = [candidates[i] for i in top]
    env.record = was_recording

    # Final selection: among survivors, pick by mean_q + gumbel score
    # (not just mean_q, which breaks ties arbitrarily when all actions score 0)
    final_scores = np.full(n_actions, -np.inf)
    for a in candidates:
        mq = total_q[a] / visits[a] if visits[a] > 0 else 0.0
        final_scores[a] = mq + gumbel[a]
    return int(np.argmax(final_scores))


def _restore(env, state):
    raw = env.env.unwrapped
    raw.agent.position = [float(state[0]), float(state[1])]
    raw.agent.velocity = (0, 0)
    raw.block.angle = float(state[4])
    raw.block.position = [float(state[2]), float(state[3])]
    raw.block.velocity = (0, 0)
    raw.block.angular_velocity = 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-macros", type=int, default=15)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--gap-size", type=int, default=200)
    p.add_argument("--no-prior", action="store_true",
                   help="Skip diffusion policy, use uniform prior")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    record = not args.no_gif

    policy = None if args.no_prior else _load_policy()

    print(f"\n{'='*60}")
    print(f" PushT Wall Gap  |  gap={args.gap_size}px  budget={args.budget}")
    print(f" seed={args.seed}  n_macros={args.n_macros}")
    print(f" prior={'uniform' if policy is None else 'diffusion'}")
    print(f"{'='*60}\n")

    print("── PUCT ──")
    np.random.seed(args.seed)
    env_p = WallGapEnv(gap_size=args.gap_size, seed=args.seed, record=record)
    fracs_p = run_episode(env_p, "puct", args.n_macros, args.budget, policy)
    frames_p = list(env_p.frames)
    env_p.close()

    print(f"\n── Gumbel ──")
    np.random.seed(args.seed + 1)
    env_g = WallGapEnv(gap_size=args.gap_size, seed=args.seed, record=record)
    fracs_g = run_episode(env_g, "gumbel", args.n_macros, args.budget, policy)
    frames_g = list(env_g.frames)
    env_g.close()

    print(f"\n{'='*60}")
    print(f"  puct    final frac: {fracs_p[-1]:.3f}  |  max: {max(fracs_p):.3f}")
    print(f"  gumbel  final frac: {fracs_g[-1]:.3f}  |  max: {max(fracs_g):.3f}")
    print(f"{'='*60}\n")

    if record and frames_p and frames_g:
        gif_path = os.path.join(out_dir, f"wall_gap_{args.gap_size}.gif")
        label_p = f"PUCT  frac={fracs_p[-1]:.2f}"
        label_g = f"Gumbel  frac={fracs_g[-1]:.2f}"
        make_gif(frames_p, frames_g, label_p, label_g, gif_path, fps=15)


if __name__ == "__main__":
    main()
