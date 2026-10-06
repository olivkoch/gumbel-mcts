"""
demo/wall_gap_lib.py — Wall-gap task using the gumbel_mcts library.

Usage:
    uv run python demo/wall_gap_lib.py --gap-size 200 --budget 32
    uv run python demo/wall_gap_lib.py --gap-size 150 --budget 64 --no-prior
"""

import argparse
import json
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
sys.path.insert(0, os.path.dirname(__file__))

from wall_gap_logic import (
    WallGapLogic, WallGapModel, PythonPUCT, PythonGumbelDense,
    _get_wall_env, _restore_state, _fraction_past_wall, _keypoints,
)
from pusht import make_gif


def run_episode(algo, logic, model, num_sims, n_macros, record=False):
    """Run one episode. Returns (fracs, frames)."""
    logic.reset()
    board = logic.get_initial_board()
    max_nodes = max(num_sims * 8 + 100, 400)

    fracs = []
    frames = []

    if record:
        import gymnasium as gym
        rec_env = gym.make("gym_pusht/PushT-v0", obs_type="state",
                           render_mode="rgb_array")
        rec_env = rec_env.env if isinstance(rec_env, gym.wrappers.TimeLimit) else rec_env
        rec_env.reset(seed=logic.seed)
        raw = rec_env.unwrapped
        raw.goal_pose = np.array([-1000.0, -1000.0, 0.0])
        raw.block.angle = 0.0
        raw.block.position = (150, 256)
        raw.block.velocity = (0, 0)
        raw.block.angular_velocity = 0
        raw.agent.position = (80, 256)
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
                                     logic=logic, device="cpu",
                                     max_considered_actions=8)
            tree.initialize_roots([0], board[None], np.array([1]))
            action = int(
                tree.run_simulation_batch(model, [0],
                                          num_simulations=num_sims)[0]
            )

        dt = time.perf_counter() - t0

        _, _, _, board = logic.fast_step(board.copy(), action, 1)
        frac = _fraction_past_wall(board)
        fracs.append(frac)


        if record:
            _restore_state(rec_env, board)
            frames.append(rec_env.render())

        print(f"[{algo:6s}] step {step+1}/{n_macros} | action={action} | "
              f"frac={frac:.3f} | plan={dt:.2f}s")

        if frac >= 0.95:
            break

    if record and 'rec_env' in dir():
        rec_env.close()

    return fracs, frames


# ── Sweep mode ───────────────────────────────────────────────────────────────

def run_sweep(args):
    seeds = [int(s * 137 + 42) for s in range(args.seeds)]
    results = {}
    total_runs = len(args.gaps) * len(args.budgets) * len(seeds) * 2
    run_idx = 0

    for gap in args.gaps:
        results[gap] = {}
        for budget in args.budgets:
            puct_fracs, gumbel_fracs = [], []

            print(f"\n{'='*60}")
            print(f" Gap={gap}  Budget={budget}  ({args.seeds} seeds)")
            print(f"{'='*60}")

            for seed in seeds:
                t0 = time.perf_counter()

                logic = WallGapLogic(gap_size=gap, seed=seed)
                model = WallGapModel(logic, use_geometric_prior=not args.no_prior)

                np.random.seed(seed); torch.manual_seed(seed)
                fp_list, _ = run_episode("puct", logic, model, budget, args.n_macros)
                fp = fp_list[-1]

                np.random.seed(seed + 1); torch.manual_seed(seed + 1)
                logic.reset(seed)
                fg_list, _ = run_episode("gumbel", logic, model, budget, args.n_macros)
                fg = fg_list[-1]

                puct_fracs.append(fp)
                gumbel_fracs.append(fg)
                run_idx += 2
                dt = time.perf_counter() - t0
                print(f"  seed={seed:5d}  puct={fp:.3f}  gumbel={fg:.3f}  "
                      f"({dt:.1f}s)  [{run_idx}/{total_runs}]")

            pa, ga = np.array(puct_fracs), np.array(gumbel_fracs)
            results[gap][budget] = {
                "puct_mean": float(np.mean(pa)),
                "puct_std": float(np.std(pa)),
                "puct_success": float(np.mean(pa >= 0.95)),
                "gumbel_mean": float(np.mean(ga)),
                "gumbel_std": float(np.std(ga)),
                "gumbel_success": float(np.mean(ga >= 0.95)),
            }
            r = results[gap][budget]
            print(f"\n  PUCT:   mean={r['puct_mean']:.3f}±{r['puct_std']:.3f}  "
                  f"success={r['puct_success']:.1%}")
            print(f"  Gumbel: mean={r['gumbel_mean']:.3f}±{r['gumbel_std']:.3f}  "
                  f"success={r['gumbel_success']:.1%}")

    out_path = os.path.join(args.results_dir, "wall_gap_lib_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*80}")
    print(f" SUMMARY — fraction past wall (mean ± std) / success rate (frac ≥ 0.95)")
    print(f"{'='*80}")
    for gap in args.gaps:
        print(f"\n  Gap = {gap}px")
        print(f"  {'Budget':>8s}  {'PUCT mean':>12s}  {'PUCT succ':>10s}  "
              f"{'Gumbel mean':>12s}  {'Gumbel succ':>11s}  {'Winner':>8s}")
        print(f"  {'-'*70}")
        for budget in args.budgets:
            r = results[gap][budget]
            pm = f"{r['puct_mean']:.3f}±{r['puct_std']:.3f}"
            ps = f"{r['puct_success']:.0%}"
            gm = f"{r['gumbel_mean']:.3f}±{r['gumbel_std']:.3f}"
            gs = f"{r['gumbel_success']:.0%}"
            winner = "Gumbel" if r['gumbel_success'] > r['puct_success'] else \
                     "PUCT" if r['puct_success'] > r['gumbel_success'] else "Tie"
            print(f"  {budget:>8d}  {pm:>12s}  {ps:>10s}  {gm:>12s}  {gs:>11s}  {winner:>8s}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-macros", type=int, default=15)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--gap-size", type=int, default=200)
    p.add_argument("--no-prior", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    # Sweep mode
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--gaps", type=int, nargs="+", default=[200, 150, 100])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--results-dir", type=str, default="demo")
    args = p.parse_args()

    if args.sweep:
        run_sweep(args)
        return

    out_dir = os.path.dirname(os.path.abspath(__file__))
    logic = WallGapLogic(gap_size=args.gap_size, seed=args.seed)
    model = WallGapModel(logic, use_geometric_prior=not args.no_prior)
    record = not args.no_gif

    prior_label = "uniform" if args.no_prior else "geometric"
    print(f"\n{'='*60}")
    print(f" Wall Gap (library)  |  gap={args.gap_size}px  budget={args.budget}")
    print(f" seed={args.seed}  n_macros={args.n_macros}  prior={prior_label}")
    print(f"{'='*60}\n")

    print("── PUCT ──")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    fracs_p, frames_p = run_episode("puct", logic, model, args.budget,
                                    args.n_macros, record)

    print(f"\n── Gumbel ──")
    np.random.seed(args.seed + 1); torch.manual_seed(args.seed + 1)
    logic.reset(args.seed)
    fracs_g, frames_g = run_episode("gumbel", logic, model, args.budget,
                                    args.n_macros, record)

    print(f"\n{'='*60}")
    print(f"  puct    final frac: {fracs_p[-1]:.3f}  |  max: {max(fracs_p):.3f}")
    print(f"  gumbel  final frac: {fracs_g[-1]:.3f}  |  max: {max(fracs_g):.3f}")
    print(f"{'='*60}\n")

    if record and frames_p and frames_g:
        gif_path = os.path.join(out_dir, f"wall_gap_lib_{args.gap_size}.gif")
        label_p = f"PUCT  frac={fracs_p[-1]:.2f}"
        label_g = f"Gumbel  frac={fracs_g[-1]:.2f}"
        make_gif(frames_p, frames_g, label_p, label_g, gif_path, fps=15)


if __name__ == "__main__":
    main()
