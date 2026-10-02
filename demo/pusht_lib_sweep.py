"""
demo/pusht_lib_sweep.py — Budget sweep using the gumbel_mcts library.

Usage:
    uv run python demo/pusht_lib_sweep.py --seeds 50 --budgets 8 16 32 64 128
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from pusht_logic import PushTLogic, PushTModel
from pusht_lib import run_episode

SUCCESS_THRESHOLD = 0.40


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--n-macros", type=int, default=10)
    args = p.parse_args()

    logic = PushTLogic()
    model = PushTModel(logic)
    seeds = [int(s * 137 + 42) for s in range(args.seeds)]
    results = {}

    t_start = time.perf_counter()

    for budget in args.budgets:
        print(f"\n{'='*60}")
        print(f" Budget = {budget}")
        print(f"{'='*60}")

        puct_finals, gumbel_finals = [], []

        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            logic.reset(seed)
            covs_p, _ = run_episode("puct", logic, model, budget, args.n_macros)

            np.random.seed(seed + 1); torch.manual_seed(seed + 1)
            logic.reset(seed)
            covs_g, _ = run_episode("gumbel", logic, model, budget, args.n_macros)

            puct_finals.append(covs_p[-1])
            gumbel_finals.append(covs_g[-1])
            print(f"  seed={seed:5d}  puct={covs_p[-1]:.3f}  gumbel={covs_g[-1]:.3f}")

        results[budget] = {"puct_finals": puct_finals, "gumbel_finals": gumbel_finals}

    elapsed = time.perf_counter() - t_start
    n = len(seeds)

    print(f"\n{'='*72}")
    print(f" PushT (library) Budget Sweep — {n} seeds, success = final IoU>{SUCCESS_THRESHOLD:.2f}")
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

    print(f"{'='*72}")
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
