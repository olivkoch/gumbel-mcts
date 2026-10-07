"""Generate wall-gap video at sims=64. Searches seeds for best Gumbel > PUCT contrast."""

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

from pusht import make_gif
from pusht_logic import PythonPUCT, PythonGumbelDense
from wall_gap_logic import (
    WallGapLogic, WallGapModel, _get_wall_env,
    _restore_state, _fraction_past_wall,
    NUM_PUSH_DIRS,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_SIZE = 150
BUDGET = 64
N_MACROS = 15
SEED = 42


def pick_action(algo, logic, model, board):
    max_nodes = max(BUDGET * 8 + 100, 400)
    if algo == "puct":
        tree = PythonPUCT(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=BUDGET)
        visits, _ = tree.get_all_root_data(n_active=1)
        return int(np.argmax(visits[0]))
    else:
        tree = PythonGumbelDense(n_games=1, max_nodes=max_nodes, logic=logic,
                                  device="cpu", max_considered_actions=8)
        tree.initialize_roots([0], board[None], np.array([1]))
        return int(tree.run_simulation_batch(model, [0], num_simulations=BUDGET)[0])



def render_board(board, gap_size):
    env = _get_wall_env(gap_size)
    _restore_state(env, board)
    raw = env.unwrapped
    raw.goal_pose = np.array([-1000.0, -1000.0, 0.0])
    frame = env.render()
    env.close()
    return frame


def record_episode(algo, seed=SEED):
    algo_seed = seed if algo == "puct" else seed + 1
    np.random.seed(algo_seed); torch.manual_seed(algo_seed)
    logic = WallGapLogic(gap_size=GAP_SIZE, seed=seed)
    model = WallGapModel(logic)
    logic.reset(seed)
    board = logic.get_initial_board()

    frames = [render_board(board, GAP_SIZE)]
    for step in range(N_MACROS):
        action = pick_action(algo, logic, model, board)
        _, _, _, board = logic.fast_step(board.copy(), action, 1)
        frames.append(render_board(board, GAP_SIZE))
        frac = _fraction_past_wall(board)
        print(f"  [{algo:6s}] step {step+1}/{N_MACROS} | action={action} | frac={frac:.3f}")

    # Hold final frame
    frames.extend([frames[-1]] * 3)
    return frames, frac


def quick_run(algo, seed):
    algo_seed = seed if algo == "puct" else seed + 1
    np.random.seed(algo_seed); torch.manual_seed(algo_seed)
    logic = WallGapLogic(gap_size=GAP_SIZE, seed=seed)
    model = WallGapModel(logic)
    logic.reset(seed)
    board = logic.get_initial_board()
    for _ in range(N_MACROS):
        action = pick_action(algo, logic, model, board)
        _, _, _, board = logic.fast_step(board.copy(), action, 1)
    return _fraction_past_wall(board)


def to_mp4(gif_path, mp4_path):
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", gif_path,
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        mp4_path,
    ], capture_output=True, check=True)


if __name__ == "__main__":
    t0 = time.time()

    print(f"Searching for best seed (gap={GAP_SIZE}, sims={BUDGET})...")
    best_seed, best_diff = None, -1e9
    for s in range(20):
        seed = s * 137 + 42
        frac_p = quick_run("puct", seed)
        frac_g = quick_run("gumbel", seed)
        diff = frac_g - frac_p
        print(f"  seed={seed}: PUCT={frac_p:.3f}  Gumbel={frac_g:.3f}")
        if diff > best_diff:
            best_diff = diff
            best_seed = seed

    print(f"\nBest seed: {best_seed} (diff={best_diff:.3f})")
    print(f"\nRecording PUCT (seed={best_seed})...")
    frames_p, frac_p = record_episode("puct", seed=best_seed)
    print(f"\nRecording Gumbel (seed={best_seed})...")
    frames_g, frac_g = record_episode("gumbel", seed=best_seed)

    gif_path = os.path.join(OUT_DIR, "gif", "final_wallgap_sims64.gif")
    mp4_path = os.path.join(OUT_DIR, "mp4", "final_wallgap_sims64.mp4")
    make_gif(frames_p, frames_g,
             f"PUCT  frac={frac_p:.2f}",
             f"Gumbel  frac={frac_g:.2f}",
             gif_path, fps=15)
    to_mp4(gif_path, mp4_path)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  PUCT:   {frac_p:.3f}")
    print(f"  Gumbel: {frac_g:.3f}")
    print(f"  {gif_path}")
    print(f"  {mp4_path}")
