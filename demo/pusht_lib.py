"""
demo/pusht_lib.py — PushT using the gumbel_mcts library (PUCT & GumbelDense).

Same experiment as pusht.py but using the library's tree search implementation
instead of a custom flat-bandit planner. This ensures the comparison uses
identical MCTS code across all demos (car parking, board games, PushT).

Usage:
    uv run python demo/pusht_lib.py
    uv run python demo/pusht_lib.py --seed 42 --budget 128 --n-macros 10
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pusht_logic import PushTLogic, PushTModel, PythonPUCT, PythonGumbelDense
from pusht import make_gif, _load_policy


def run_episode(algo, logic, model, num_sims, n_macros, record=False):
    """Run one episode. Returns (coverages, frames)."""
    logic.reset()
    model.reset_obs_history()
    board = logic.get_initial_board()
    max_nodes = max(num_sims * 8 + 100, 400)

    coverages = []
    frames = []

    if record:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
        rec_env = gym.make("gym_pusht/PushT-v0", obs_type="state",
                           render_mode="rgb_array")
        rec_env = rec_env.env if isinstance(rec_env, gym.wrappers.TimeLimit) else rec_env
        rec_env.reset(seed=logic.seed)
        frames.append(rec_env.render())

    for step in range(n_macros):
        t0 = time.perf_counter()

        if algo == "puct":
            tree = PythonPUCT(n_games=1, max_nodes=max_nodes,
                              logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            tree.run_simulation_batch(model, [0], num_simulations=num_sims)
            visits, _ = tree.get_all_root_data(n_active=1)
            action = int(np.argmax(visits[0]))
        else:
            tree = PythonGumbelDense(n_games=1, max_nodes=max_nodes,
                                     logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            action = int(
                tree.run_simulation_batch(model, [0],
                                          num_simulations=num_sims)[0]
            )

        dt = time.perf_counter() - t0

        # Execute the chosen action on the actual board
        _, _, _, board = logic.fast_step(board.copy(), action, 1)

        # Get IoU from the environment state
        from pusht_logic import _get_env, _restore_state, _keypoints
        env = _get_env()
        _restore_state(env, board)
        raw = env.unwrapped
        cov = raw._get_coverage()
        coverages.append(cov)

        # Update obs history for diffusion prior
        model._obs_history.append({
            "environment_state": _keypoints(raw.block).flatten().astype(np.float32),
            "agent_pos": np.array(raw.agent.position, dtype=np.float32),
        })

        if record:
            from pusht_logic import _restore_state as rs
            rs(rec_env, board)
            frames.append(rec_env.render())

        print(f"[{algo:6s}] step {step+1}/{n_macros} | action={action} | "
              f"IoU={cov:.3f} | plan={dt:.2f}s")

        if cov >= 0.80:
            break

    if record and 'rec_env' in dir():
        rec_env.close()

    return coverages, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-macros", type=int, default=10)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--no-prior", action="store_true",
                   help="Skip diffusion policy, use uniform prior")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    policy = None if args.no_prior else _load_policy()
    logic = PushTLogic(seed=args.seed)
    model = PushTModel(logic, diffusion_policy=policy)

    prior_label = "uniform" if policy is None else "diffusion"
    print(f"\n{'='*60}")
    print(f" PushT (library)  —  PUCT vs Gumbel  |  budget={args.budget}")
    print(f" seed={args.seed}  n_macros={args.n_macros}  prior={prior_label}")
    print(f"{'='*60}\n")

    record = not args.no_gif

    print(f"── PUCT ──")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    covs_p, frames_p = run_episode("puct", logic, model, args.budget,
                                   args.n_macros, record)

    print(f"\n── Gumbel ──")
    np.random.seed(args.seed + 1); torch.manual_seed(args.seed + 1)
    logic.reset(args.seed)
    covs_g, frames_g = run_episode("gumbel", logic, model, args.budget,
                                   args.n_macros, record)

    print(f"\n{'='*60}")
    print(f"  puct    final IoU: {covs_p[-1]:.3f}  |  mean: {np.mean(covs_p):.3f}")
    print(f"  gumbel  final IoU: {covs_g[-1]:.3f}  |  mean: {np.mean(covs_g):.3f}")
    print(f"{'='*60}\n")

    if record:
        gif_path = os.path.join(out_dir, "gif", "pusht_lib_comparison.gif")
        label_p = f"PUCT  IoU={covs_p[-1]:.2f}"
        label_g = f"Gumbel  IoU={covs_g[-1]:.2f}"
        make_gif(frames_p, frames_g, label_p, label_g, gif_path, fps=15)


if __name__ == "__main__":
    main()
