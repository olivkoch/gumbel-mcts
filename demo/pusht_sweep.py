"""
demo/pusht_sweep.py — Budget sweep: PUCT vs Gumbel on PushT.

Runs multiple seeds per simulation budget, reports success rates and
mean IoU, and generates a side-by-side GIF for each budget.

Usage:
    uv run python demo/pusht_sweep.py
    uv run python demo/pusht_sweep.py --seeds 20 --budgets 16 32 64 128 256
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pusht import (
    _load_policy, _compute_prior, _make_obs,
    run_episode, make_gif,
)

SUCCESS_THRESHOLD = 0.40


def run_sweep(budgets, seeds, n_macros, policy, out_dir, rollout_depth=1):
    results = {}

    for budget in budgets:
        print(f"\n{'='*60}")
        print(f" Budget = {budget}")
        print(f"{'='*60}")

        puct_finals, gumbel_finals = [], []
        gif_frames_puct, gif_frames_gumbel = None, None

        for i, seed in enumerate(seeds):
            record = (i == 0)
            t0 = time.perf_counter()

            covs_p, frames_p = run_episode(
                "puct", seed, n_macros, budget, policy, record,
                rollout_depth=rollout_depth,
            )
            covs_g, frames_g = run_episode(
                "gumbel", seed, n_macros, budget, policy, record,
                rollout_depth=rollout_depth,
            )

            dt = time.perf_counter() - t0
            puct_finals.append(covs_p[-1])
            gumbel_finals.append(covs_g[-1])

            if record:
                gif_frames_puct = frames_p
                gif_frames_gumbel = frames_g

            print(f"  seed={seed:3d}  puct={covs_p[-1]:.3f}  gumbel={covs_g[-1]:.3f}  ({dt:.1f}s)")

        results[budget] = {
            "puct_finals": puct_finals,
            "gumbel_finals": gumbel_finals,
        }

        if gif_frames_puct and gif_frames_gumbel:
            gif_path = os.path.join(out_dir, "gif", f"pusht_budget{budget}.gif")
            make_gif(
                gif_frames_puct, gif_frames_gumbel,
                f"PUCT (n={budget})", f"Gumbel (n={budget})",
                gif_path, fps=15,
            )

    return results


def print_summary(results, seeds):
    n = len(seeds)
    print(f"\n{'='*72}")
    print(f" PushT Budget Sweep — {n} seeds, success = final IoU>{SUCCESS_THRESHOLD:.2f}")
    print(f"{'='*72}")
    print(f"{'Budget':>8s}  |  {'PUCT success':>13s}  {'mean final':>11s}  |  {'Gumbel success':>14s}  {'mean final':>11s}")
    print(f"{'-'*8}--+--{'-'*13}--{'-'*11}--+--{'-'*14}--{'-'*11}")

    for budget in sorted(results):
        r = results[budget]
        p_succ = sum(1 for v in r["puct_finals"] if v >= SUCCESS_THRESHOLD) / n
        g_succ = sum(1 for v in r["gumbel_finals"] if v >= SUCCESS_THRESHOLD) / n
        p_mean = np.mean(r["puct_finals"])
        g_mean = np.mean(r["gumbel_finals"])
        print(f"{budget:>8d}  |  {p_succ:>12.0%}  {p_mean:>11.3f}  |  {g_succ:>13.0%}  {g_mean:>11.3f}")

    print(f"{'='*72}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    p.add_argument("--n-macros", type=int, default=10)
    p.add_argument("--rollout-depth", type=int, default=1,
                   help="Lookahead depth per simulation (receding horizon)")
    p.add_argument("--no-prior", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    seeds = [int(s * 137 + 42) for s in range(args.seeds)]

    policy = None if args.no_prior else _load_policy()

    t_start = time.perf_counter()
    results = run_sweep(args.budgets, seeds, args.n_macros, policy, out_dir,
                        rollout_depth=args.rollout_depth)
    elapsed = time.perf_counter() - t_start

    print_summary(results, seeds)
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
