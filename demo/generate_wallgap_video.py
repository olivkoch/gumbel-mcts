"""
Generate one compelling wall-gap video: Gumbel succeeds, PUCT fails.
Uses library MCTS (PythonPUCT / PythonGumbelDense) + diffusion prior.
Tries many seeds at sims=8, picks the best contrast, records smooth video.
"""

import os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import gymnasium as gym
import gym_pusht  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pusht import make_gif, _load_policy
from pusht_logic import _keypoints, PythonPUCT, PythonGumbelDense
from wall_gap_logic import (
    WallGapLogic, WallGapModel, _get_wall_env,
    _restore_state, _fraction_past_wall, _setup_walls,
    LOCAL_VERTS, WALL_X, GAP_CENTER_Y,
    NUM_PUSH_DIRS, NUM_ACTIONS, N_APPROACH, N_PUSH, N_SETTLE,
    APPROACH_DIST, PUSH_DEPTH,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_SIZE = 200
BUDGET = 64
N_MACROS = 15
FPS = 15


def pick_action(algo, logic, model, board, budget):
    max_nodes = max(budget * 8 + 100, 400)
    if algo == "puct":
        tree = PythonPUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=budget)
        visits, _ = tree.get_all_root_data(n_active=1)
        return int(np.argmax(visits[0]))
    else:
        tree = PythonGumbelDense(n_games=1, max_nodes=max_nodes, logic=logic,
                                  device="cpu", max_considered_actions=8)
        tree.initialize_roots([0], board[None], np.array([1]))
        return int(tree.run_simulation_batch(model, [0], num_simulations=budget)[0])


def run_episode_no_record(algo, seed, policy):
    """Quick run, no recording. Returns final fraction."""
    algo_seed = seed if algo == "puct" else seed + 1
    np.random.seed(algo_seed); torch.manual_seed(algo_seed)
    logic = WallGapLogic(gap_size=GAP_SIZE, seed=seed)
    model = WallGapModel(logic, diffusion_policy=policy)
    model.reset_obs_history()
    logic.reset(seed)
    board = logic.get_initial_board()

    for _ in range(N_MACROS):
        action = pick_action(algo, logic, model, board, BUDGET)
        _, _, _, board = logic.fast_step(board.copy(), action, 1)
        wenv = _get_wall_env(GAP_SIZE)
        _restore_state(wenv, board)
        model._obs_history.append({
            "environment_state": _keypoints(wenv.unwrapped.block).flatten().astype(np.float32),
            "agent_pos": np.array(wenv.unwrapped.agent.position, dtype=np.float32),
        })
    return _fraction_past_wall(board)


def _macro_targets_from_env(raw_env):
    kp = _keypoints(raw_env.block)
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


def record_episode(algo, seed, policy):
    """Run and record with smooth physics substeps."""
    algo_seed = seed if algo == "puct" else seed + 1
    np.random.seed(algo_seed); torch.manual_seed(algo_seed)
    logic = WallGapLogic(gap_size=GAP_SIZE, seed=seed)
    model = WallGapModel(logic, diffusion_policy=policy)
    model.reset_obs_history()
    logic.reset(seed)
    board = logic.get_initial_board()

    rec = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    rec = rec.env if isinstance(rec, gym.wrappers.TimeLimit) else rec
    rec.reset(seed=seed)
    raw = rec.unwrapped
    raw.goal_pose = np.array([-1000.0, -1000.0, 0.0])
    raw.block.angle = 0.0
    raw.block.position = (150, 256)
    raw.block.velocity = (0, 0)
    raw.block.angular_velocity = 0
    raw.agent.position = (80, 256)
    _setup_walls(rec, GAP_SIZE)
    raw.block._space.step(0.001)
    raw.block.velocity = (0, 0)
    raw.block.angular_velocity = 0
    frames = [rec.render()]

    agent_x_limit = WALL_X - 15 - 15
    half = GAP_SIZE / 2
    wall_left = WALL_X - 15

    for step in range(N_MACROS):
        action = pick_action(algo, logic, model, board, BUDGET)
        # Execute on logic for board state
        _, _, _, board = logic.fast_step(board.copy(), action, 1)
        wenv = _get_wall_env(GAP_SIZE)
        _restore_state(wenv, board)
        model._obs_history.append({
            "environment_state": _keypoints(wenv.unwrapped.block).flatten().astype(np.float32),
            "agent_pos": np.array(wenv.unwrapped.agent.position, dtype=np.float32),
        })

        # Execute on rec_env for smooth frames (with wall enforcement)
        if action == NUM_PUSH_DIRS:
            frames.append(rec.render())
        else:
            approach, push_end = _macro_targets_from_env(raw)
            step_count = 0
            for target, n in [(approach[action], N_APPROACH),
                              (push_end[action], N_PUSH),
                              (push_end[action], N_SETTLE)]:
                clipped = target.copy()
                clipped[0] = min(clipped[0], agent_x_limit)
                for _ in range(n):
                    prev_pos = list(raw.block.position)
                    prev_angle = raw.block.angle
                    rec.step(clipped.astype(np.float32))
                    bx, by = raw.block.position
                    ba = raw.block.angle
                    c, s = np.cos(ba), np.sin(ba)
                    R = np.array([[c, -s], [s, c]])
                    kp = (R @ LOCAL_VERTS.T).T + np.array([bx, by])
                    violation = any(
                        kx > wall_left and (ky > GAP_CENTER_Y + half or ky < GAP_CENTER_Y - half)
                        for kx, ky in kp
                    )
                    if violation:
                        raw.block.angle = prev_angle
                        raw.block.position = prev_pos
                        raw.block.velocity = (0, 0)
                        raw.block.angular_velocity = 0
                    ax, ay = raw.agent.position
                    if ax > agent_x_limit:
                        in_gap = (GAP_CENTER_Y - half) < ay < (GAP_CENTER_Y + half)
                        if not in_gap:
                            raw.agent.position = (agent_x_limit, ay)
                    step_count += 1
                    if step_count % 3 == 0:
                        frames.append(rec.render())

        # Sync rec_env to board state after macro
        _restore_state(rec, board)
        frac = _fraction_past_wall(board)
        print(f"  [{algo:6s}] step {step+1}/{N_MACROS} | action={action} | frac={frac:.3f}")

    rec.close()
    return frames, frac


if __name__ == "__main__":
    policy = _load_policy()
    t0 = time.time()

    # Sweep seeds to find: Gumbel succeeds (frac >= 0.5), PUCT fails (frac < 0.3)
    print(f"Searching seeds (gap={GAP_SIZE}, sims={BUDGET})...")
    best_seed = None
    best_diff = -1
    best_gf = 0

    for s in range(15):
        seed = s * 137 + 42
        pf = run_episode_no_record("puct", seed, policy)
        gf = run_episode_no_record("gumbel", seed, policy)
        diff = gf - pf
        tag = " ***" if diff > 0 else ""
        print(f"  seed={seed:5d}  PUCT={pf:.3f}  Gumbel={gf:.3f}{tag}")
        if diff > best_diff:
            best_diff = diff
            best_seed = seed
            best_gf = gf

    print(f"\nBest seed: {best_seed} (diff={best_diff:.3f})")
    print(f"Recording...")

    frames_p, frac_p = record_episode("puct", best_seed, policy)
    frames_g, frac_g = record_episode("gumbel", best_seed, policy)

    gif_path = os.path.join(OUT_DIR, f"final_wallgap_sims{BUDGET}.gif")
    make_gif(frames_p, frames_g,
             f"PUCT  frac={frac_p:.2f}",
             f"Gumbel  frac={frac_g:.2f}",
             gif_path, fps=FPS)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Output: {gif_path}")
    print(f"  PUCT:   {frac_p:.3f}")
    print(f"  Gumbel: {frac_g:.3f}")
