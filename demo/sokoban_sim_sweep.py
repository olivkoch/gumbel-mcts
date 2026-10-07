"""
demo/sokoban_sim_sweep.py — Simulation-budget sweep: Gumbel vs PUCT.

Trains both algorithms at sims ∈ {8, 16, 32, 64} with BFS warmup and plots
win rate vs training episodes for each budget level.

This answers: at which simulation budget does Gumbel's sequential halving
outperform PUCT's UCB?

Usage
-----
    uv run python demo/sokoban_sim_sweep.py
"""

import argparse
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Import shared infrastructure from sokoban_train without running main().
import importlib.util, pathlib
_mod_path = pathlib.Path(__file__).parent / "sokoban_train.py"
_spec = importlib.util.spec_from_file_location("sokoban_train", _mod_path)
_st = importlib.util.module_from_spec(_spec)
import sys as _sys; _sys.modules["sokoban_train"] = _st; _spec.loader.exec_module(_st)
from sokoban_train import (
    SokobanGame, parse_puzzle, train_system,
    TRAIN_TEXTS, EVAL_TEXT,
)


def make_args(sims, warmup=30, iters=40, episodes_per_iter=5,
              train_steps=20, eval_every=5, n_eval=20,
              replay_size=8000, pretrain_steps=200, seed=0):
    """Build a simple namespace that train_system expects."""
    a = types.SimpleNamespace()
    a.sims            = sims
    a.sims_warmup     = 256
    a.sims_eval       = sims * 2        # eval at 2× training budget
    a.warmup          = warmup
    a.iters           = iters
    a.episodes_per_iter = episodes_per_iter
    a.train_steps     = train_steps
    a.eval_every      = eval_every
    a.n_eval          = n_eval
    a.replay_size     = replay_size
    a.pretrain_steps  = pretrain_steps
    a.min_wins        = 0
    a.seed            = seed
    return a


def plot_sweep(results, out_path):
    """results: list of (sims, gumbel_curve, puct_curve)"""
    ncols = 2
    nrows = (len(results) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, 4 * nrows),
                             sharey=True)
    axes = np.array(axes).flatten()

    for ax, (sims, g_curve, p_curve) in zip(axes, results):
        for label, curve, color, marker in [
            ("Gumbel", g_curve, "#26A69A", "s"),
            ("PUCT",   p_curve, "#5C6BC0", "o"),
        ]:
            ep = [c[0] for c in curve]
            sr = [c[1] for c in curve]
            ax.plot(ep, sr, f"{marker}-", color=color, label=label,
                    lw=2, ms=6, mfc="white", mew=2)
        ax.set_title(f"sims = {sims}", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.axhline(50, color="#BDBDBD", ls="--", lw=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.supxlabel("Self-play training episodes", fontsize=11, y=0.02)
    fig.supylabel("Train win rate (%)", fontsize=11, x=0.02)
    fig.suptitle(
        "Gumbel vs PUCT: training win rate across simulation budgets\n"
        "(BFS warmup, macro-push actions, Microban #1/#20/#21)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets",    type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--warmup",     type=int, default=30)
    ap.add_argument("--iters",      type=int, default=40)
    ap.add_argument("--ep-per-iter", type=int, default=5)
    ap.add_argument("--train-steps", type=int, default=20)
    ap.add_argument("--n-eval",     type=int, default=20)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out",        type=str,
                    default="demo/png/sokoban_sim_sweep.png")
    args = ap.parse_args()

    print("Sokoban — simulation budget sweep: Gumbel vs PUCT")
    print(f"  Budgets: {args.budgets}  |  warmup={args.warmup}  iters={args.iters}")
    print("  Parsing puzzles and compiling Numba kernels ...")

    train_puzzles = [parse_puzzle(t, f"Microban #{n}")
                     for t, n in zip(TRAIN_TEXTS, [1, 20, 21])]
    eval_puzzle   = parse_puzzle(EVAL_TEXT, "Microban #19")
    train_games   = [SokobanGame(p) for p in train_puzzles]
    eval_game     = SokobanGame(eval_puzzle)

    # JIT warm-up
    import numpy as _np
    dummy = _np.zeros(2 + 2 * 2, dtype=_np.int8)
    for g in train_games + [eval_game]:
        g.fast_step(dummy.copy(), 0, 1)
        g.get_valid_mask(dummy, 1)
    print("  Kernels ready.\n")

    group = f"sweep-seed{args.seed}"
    results = []

    for sims in args.budgets:
        print(f"\n{'='*60}")
        print(f"  Budget: sims={sims}")
        print(f"{'='*60}")
        run_args = make_args(
            sims=sims,
            warmup=args.warmup,
            iters=args.iters,
            episodes_per_iter=args.ep_per_iter,
            train_steps=args.train_steps,
            eval_every=5,
            n_eval=0,             # skip eval — eval puzzle always 0%, not informative
            pretrain_steps=200,   # reduced from 1000 for sweep speed
            seed=args.seed,
        )

        _, g_curve = train_system(
            "gumbel", run_args, train_games, eval_game,
            seed=args.seed,
            group=f"{group}-sims{sims}",
        )
        _, p_curve = train_system(
            "puct", run_args, train_games, eval_game,
            seed=args.seed + 1000,
            group=f"{group}-sims{sims}",
        )
        results.append((sims, g_curve, p_curve))

    plot_sweep(results, args.out)

    with wandb.init(entity="mistral-ai", project="mcts",
                    name="sweep-summary", group=group, reinit=True):
        wandb.log({"sim_sweep": wandb.Image(args.out)})


if __name__ == "__main__":
    main()
