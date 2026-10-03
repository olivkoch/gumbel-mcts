"""
demo/wall_gap_sweep.py — Budget × gap-size sweep for wall-gap PushT task.

Compares PUCT vs Gumbel (flat bandit) across simulation budgets and gap sizes.
Reports fraction_past_wall as the success metric.

Usage:
    python demo/wall_gap_sweep.py --budgets 4 8 16 32 64 128 256 \
        --gaps 100 150 200 --seeds 30 --n-macros 15
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

sys.path.insert(0, os.path.dirname(__file__))
from pusht_wall import WallGapEnv, run_episode
from pusht import _load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--gaps", type=int, nargs="+", default=[100, 150, 200])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n-macros", type=int, default=15)
    p.add_argument("--no-prior", action="store_true",
                   help="Skip diffusion policy, use uniform prior")
    p.add_argument("--results-dir", type=str, default="demo")
    args = p.parse_args()

    policy = None if args.no_prior else _load_policy()

    seeds = [int(s * 137 + 42) for s in range(args.seeds)]
    results = {}

    total_runs = len(args.gaps) * len(args.budgets) * len(seeds) * 2
    run_idx = 0

    for gap in args.gaps:
        results[gap] = {}
        for budget in args.budgets:
            key = f"gap{gap}_budget{budget}"
            puct_fracs = []
            gumbel_fracs = []

            print(f"\n{'='*60}")
            print(f" Gap={gap}  Budget={budget}  ({args.seeds} seeds)")
            print(f"{'='*60}")

            for si, seed in enumerate(seeds):
                t0 = time.perf_counter()

                # PUCT
                np.random.seed(seed)
                env_p = WallGapEnv(gap_size=gap, seed=seed, record=False)
                fracs_p = run_episode(env_p, "puct", args.n_macros, budget, policy)
                fp = fracs_p[-1]
                env_p.close()

                # Gumbel
                np.random.seed(seed + 1)
                env_g = WallGapEnv(gap_size=gap, seed=seed, record=False)
                fracs_g = run_episode(env_g, "gumbel", args.n_macros, budget, policy)
                fg = fracs_g[-1]
                env_g.close()

                puct_fracs.append(fp)
                gumbel_fracs.append(fg)

                run_idx += 2
                dt = time.perf_counter() - t0
                print(f"  seed={seed:5d}  puct={fp:.3f}  gumbel={fg:.3f}  "
                      f"({dt:.1f}s)  [{run_idx}/{total_runs}]")

            puct_arr = np.array(puct_fracs)
            gumbel_arr = np.array(gumbel_fracs)

            results[gap][budget] = {
                "puct_mean": float(np.mean(puct_arr)),
                "puct_std": float(np.std(puct_arr)),
                "puct_success": float(np.mean(puct_arr >= 0.5)),
                "gumbel_mean": float(np.mean(gumbel_arr)),
                "gumbel_std": float(np.std(gumbel_arr)),
                "gumbel_success": float(np.mean(gumbel_arr >= 0.5)),
            }

            r = results[gap][budget]
            print(f"\n  PUCT:   mean={r['puct_mean']:.3f}±{r['puct_std']:.3f}  "
                  f"success={r['puct_success']:.1%}")
            print(f"  Gumbel: mean={r['gumbel_mean']:.3f}±{r['gumbel_std']:.3f}  "
                  f"success={r['gumbel_success']:.1%}")

    # Save results
    out_path = os.path.join(args.results_dir, "wall_gap_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f" SUMMARY — fraction past wall (mean ± std) / success rate (frac ≥ 0.5)")
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


if __name__ == "__main__":
    main()
