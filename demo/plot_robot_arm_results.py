"""Plot robot arm sweep results: success rate vs budget with CI."""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def bootstrap_ci(successes, n, n_bootstrap=2000, ci=0.95):
    rng = np.random.default_rng(42)
    arr = np.array([1.0] * successes + [0.0] * (n - successes))
    means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=os.path.join(OUT_DIR, "json", "robot_arm_sweep_results.json"))
    p.add_argument("--output", type=str, default=os.path.join(OUT_DIR, "png", "final_plot_robot_arm.png"))
    args = p.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    budgets = sorted(int(k) for k in results.keys())

    puct_rates, gumbel_rates = [], []
    puct_lo, puct_hi = [], []
    gumbel_lo, gumbel_hi = [], []

    for b in budgets:
        r = results[str(b)]
        n = r["n_seeds"]
        pr = r["puct_success"]
        gr = r["gumbel_success"]
        puct_rates.append(pr * 100)
        gumbel_rates.append(gr * 100)

        p_lo, p_hi = bootstrap_ci(int(round(pr * n)), n)
        g_lo, g_hi = bootstrap_ci(int(round(gr * n)), n)
        puct_lo.append(p_lo * 100)
        puct_hi.append(p_hi * 100)
        gumbel_lo.append(g_lo * 100)
        gumbel_hi.append(g_hi * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(budgets))

    ax.errorbar(x, puct_rates,
                yerr=[np.array(puct_rates) - puct_lo, np.array(puct_hi) - puct_rates],
                fmt='-o', color='#4A90D9', label='PUCT', capsize=4, markersize=8, linewidth=2)
    ax.errorbar(x, gumbel_rates,
                yerr=[np.array(gumbel_rates) - gumbel_lo, np.array(gumbel_hi) - gumbel_rates],
                fmt='-s', color='#2CA02C', label='Gumbel', capsize=4, markersize=8, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Simulation Budget", fontsize=12)
    ax.set_ylabel("Success Rate (dist < 15px) %", fontsize=12)
    n_seeds = results[str(budgets[0])]["n_seeds"]
    ax.set_title(f"Robot Arm (4 joints, 2 obstacles)\n{n_seeds} seeds, randomized init", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
