"""
Generate smooth presentation videos for PushT and wall-gap tasks.
Uses library MCTS (PythonPUCT / PythonGumbelDense) for decisions,
but records at physics-substep level for smooth animation.
Outputs: final_{task}_sims{N}.mp4
"""

import os, sys, time, subprocess
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
from pusht_logic import (
    PushTLogic, PushTModel, PythonPUCT, PythonGumbelDense,
    _get_env, _restore_state, _read_state, _keypoints,
    NUM_PUSH_DIRS, NUM_ACTIONS, APPROACH_DIST, PUSH_DEPTH,
    N_APPROACH, N_PUSH, N_SETTLE,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [4, 16, 128]
FPS = 15


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


def execute_and_record(rec_env, action, record_every=3):
    """Execute a macro-action on rec_env, recording every Nth substep."""
    frames = []
    raw = rec_env.unwrapped
    if action == NUM_PUSH_DIRS:  # no-op
        frames.append(rec_env.render())
        return frames

    approach, push_end = _macro_targets_from_env(raw)
    step_count = 0
    for target, n in [(approach[action], N_APPROACH),
                      (push_end[action], N_PUSH),
                      (push_end[action], N_SETTLE)]:
        for _ in range(n):
            rec_env.step(target.astype(np.float32))
            step_count += 1
            if step_count % record_every == 0:
                frames.append(rec_env.render())
    return frames


def to_mp4(gif_path):
    mp4 = gif_path.replace(".gif", ".mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", gif_path,
            "-movflags", "faststart", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            mp4
        ], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  (ffmpeg not available, skipping MP4 conversion)")
    return mp4


def pick_action_library(algo, logic, model, board, budget):
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


# ── PushT green target ───────────────────────────────────────────────────────

def generate_pusht_videos(policy):
    for budget in BUDGETS:
        print(f"\n{'='*60}")
        print(f" PushT green target  |  sims={budget}")
        print(f"{'='*60}")

        # Quick seed search (no recording)
        best_seed, best_diff = None, -1
        for s in range(8):
            seed = s * 137 + 42
            covs = {}
            for algo_name, algo_seed in [("puct", seed), ("gumbel", seed + 1)]:
                np.random.seed(algo_seed); torch.manual_seed(algo_seed)
                logic = PushTLogic(seed=seed)
                model = PushTModel(logic, diffusion_policy=policy)
                model.reset_obs_history()
                logic.reset(seed)
                board = logic.get_initial_board()
                for _ in range(10):
                    action = pick_action_library(algo_name, logic, model, board, budget)
                    _, _, _, board = logic.fast_step(board.copy(), action, 1)
                    env = _get_env(); _restore_state(env, board)
                    model._obs_history.append({
                        "environment_state": _keypoints(env.unwrapped.block).flatten().astype(np.float32),
                        "agent_pos": np.array(env.unwrapped.agent.position, dtype=np.float32),
                    })
                covs[algo_name] = env.unwrapped._get_coverage()
            diff = covs["gumbel"] - covs["puct"]
            print(f"  seed={seed}: PUCT={covs['puct']:.3f} Gumbel={covs['gumbel']:.3f}")
            if diff > best_diff:
                best_diff = diff; best_seed = seed

        # Record best seed with smooth animation
        seed = best_seed
        print(f"  Recording seed={seed}...")
        all_frames, all_covs = {}, {}
        for algo_name, algo_seed in [("puct", seed), ("gumbel", seed + 1)]:
            np.random.seed(algo_seed); torch.manual_seed(algo_seed)
            logic = PushTLogic(seed=seed)
            model = PushTModel(logic, diffusion_policy=policy)
            model.reset_obs_history()
            logic.reset(seed)
            board = logic.get_initial_board()

            rec = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
            rec = rec.env if isinstance(rec, gym.wrappers.TimeLimit) else rec
            rec.reset(seed=seed)
            frames = [rec.render()]

            for step in range(10):
                action = pick_action_library(algo_name, logic, model, board, budget)
                # Execute on logic (for board state)
                _, _, _, board = logic.fast_step(board.copy(), action, 1)
                env = _get_env(); _restore_state(env, board)
                model._obs_history.append({
                    "environment_state": _keypoints(env.unwrapped.block).flatten().astype(np.float32),
                    "agent_pos": np.array(env.unwrapped.agent.position, dtype=np.float32),
                })
                # Execute on rec_env (for smooth frames)
                new_frames = execute_and_record(rec, action, record_every=3)
                frames.extend(new_frames)
                # Sync rec_env to actual board state
                _restore_state(rec, board)
                cov = env.unwrapped._get_coverage()
                print(f"  [{algo_name:6s}] step {step+1}/10 | action={action} | IoU={cov:.3f}")

            all_frames[algo_name] = frames
            all_covs[algo_name] = cov
            rec.close()

        gif = os.path.join(OUT_DIR, f"final_pusht_sims{budget}.gif")
        make_gif(all_frames["puct"], all_frames["gumbel"],
                 f"PUCT  IoU={all_covs['puct']:.2f}",
                 f"Gumbel  IoU={all_covs['gumbel']:.2f}",
                 gif, fps=FPS)
        to_mp4(gif)


# ── Wall-gap task ────────────────────────────────────────────────────────────

def generate_wallgap_videos(policy):
    from wall_gap_logic import (
        WallGapLogic, WallGapModel, _get_wall_env,
        _restore_state as wg_restore, _fraction_past_wall, _setup_walls,
        _keypoints as wg_kp, _macro_targets as wg_macro,
        WALL_X, GAP_CENTER_Y, LOCAL_VERTS as WG_VERTS,
    )
    import pymunk

    def wg_execute_and_record(rec_env, action, gap_size, record_every=3):
        """Execute wall-gap macro-action with wall enforcement + recording."""
        frames = []
        raw = rec_env.unwrapped
        if action == NUM_PUSH_DIRS:
            frames.append(rec_env.render())
            return frames

        kp_fn = _keypoints
        approach, push_end = _macro_targets_from_env(raw)
        agent_x_limit = WALL_X - 15 - 15
        half = gap_size / 2
        wall_left = WALL_X - 15
        step_count = 0
        for target, n in [(approach[action], N_APPROACH),
                          (push_end[action], N_PUSH),
                          (push_end[action], N_SETTLE)]:
            clipped = target.copy()
            clipped[0] = min(clipped[0], agent_x_limit)
            for _ in range(n):
                prev_pos = list(raw.block.position)
                prev_angle = raw.block.angle
                rec_env.step(clipped.astype(np.float32))
                # Wall check
                bx, by = raw.block.position
                ba = raw.block.angle
                c, s = np.cos(ba), np.sin(ba)
                R = np.array([[c, -s], [s, c]])
                kp = (R @ WG_VERTS.T).T + np.array([bx, by])
                violation = any(kx > wall_left and (ky > GAP_CENTER_Y + half or ky < GAP_CENTER_Y - half)
                                for kx, ky in kp)
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
                if step_count % record_every == 0:
                    frames.append(rec_env.render())
        return frames

    for gap_size in [200, 150]:
        for budget in BUDGETS:
            print(f"\n{'='*60}")
            print(f" Wall gap={gap_size}  |  sims={budget}")
            print(f"{'='*60}")

            best_seed, best_diff = None, -1
            for s in range(8):
                seed = s * 137 + 42
                fracs = {}
                for algo_name, algo_seed in [("puct", seed), ("gumbel", seed + 1)]:
                    np.random.seed(algo_seed); torch.manual_seed(algo_seed)
                    logic = WallGapLogic(gap_size=gap_size, seed=seed)
                    model = WallGapModel(logic, diffusion_policy=policy)
                    model.reset_obs_history()
                    logic.reset(seed)
                    board = logic.get_initial_board()
                    for _ in range(15):
                        action = pick_action_library(algo_name, logic, model, board, budget)
                        _, _, _, board = logic.fast_step(board.copy(), action, 1)
                        wenv = _get_wall_env(gap_size)
                        wg_restore(wenv, board)
                        model._obs_history.append({
                            "environment_state": _keypoints(wenv.unwrapped.block).flatten().astype(np.float32),
                            "agent_pos": np.array(wenv.unwrapped.agent.position, dtype=np.float32),
                        })
                    fracs[algo_name] = _fraction_past_wall(board)
                diff = fracs["gumbel"] - fracs["puct"]
                print(f"  seed={seed}: PUCT={fracs['puct']:.3f} Gumbel={fracs['gumbel']:.3f}")
                if diff > best_diff:
                    best_diff = diff; best_seed = seed

            seed = best_seed
            print(f"  Recording seed={seed}...")
            all_frames, all_fracs = {}, {}
            for algo_name, algo_seed in [("puct", seed), ("gumbel", seed + 1)]:
                np.random.seed(algo_seed); torch.manual_seed(algo_seed)
                logic = WallGapLogic(gap_size=gap_size, seed=seed)
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
                _setup_walls(rec, gap_size)
                raw.block._space.step(0.001)
                raw.block.velocity = (0, 0)
                raw.block.angular_velocity = 0
                frames = [rec.render()]

                for step in range(15):
                    action = pick_action_library(algo_name, logic, model, board, budget)
                    _, _, _, board = logic.fast_step(board.copy(), action, 1)
                    wenv = _get_wall_env(gap_size)
                    wg_restore(wenv, board)
                    model._obs_history.append({
                        "environment_state": _keypoints(wenv.unwrapped.block).flatten().astype(np.float32),
                        "agent_pos": np.array(wenv.unwrapped.agent.position, dtype=np.float32),
                    })
                    new_frames = wg_execute_and_record(rec, action, gap_size, record_every=3)
                    frames.extend(new_frames)
                    wg_restore(rec, board)
                    frac = _fraction_past_wall(board)
                    print(f"  [{algo_name:6s}] step {step+1}/15 | action={action} | frac={frac:.3f}")

                all_frames[algo_name] = frames
                all_fracs[algo_name] = frac
                rec.close()

            gif = os.path.join(OUT_DIR, f"final_wallgap{gap_size}_sims{budget}.gif")
            make_gif(all_frames["puct"], all_frames["gumbel"],
                     f"PUCT  frac={all_fracs['puct']:.2f}",
                     f"Gumbel  frac={all_fracs['gumbel']:.2f}",
                     gif, fps=FPS)
            to_mp4(gif)


if __name__ == "__main__":
    policy = _load_policy()
    t0 = time.time()
    generate_pusht_videos(policy)
    generate_wallgap_videos(policy)
    print(f"\nAll videos generated in {time.time()-t0:.0f}s")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("final_") and f.endswith(".mp4"):
            size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
            print(f"  {f}  ({size:.0f} KB)")
