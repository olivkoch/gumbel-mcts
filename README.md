[![PyPI version](https://badge.fury.io/py/gumbel-mcts.svg)](https://pypi.org/project/gumbel-mcts/)
[![Tests](https://github.com/olivkoch/gumbel-mcts/actions/workflows/tests.yml/badge.svg)](https://github.com/olivkoch/gumbel-mcts/actions)
[![codecov](https://codecov.io/gh/olivkoch/gumbel-mcts/branch/main/graph/badge.svg)](https://codecov.io/gh/olivkoch/gumbel-mcts)
[![docs](https://readthedocs.org/projects/gumbel-mcts/badge/?version=latest)](https://gumbel-mcts.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/gumbel-mcts/month)](https://pepy.tech/project/gumbel-mcts)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# gumbel-mcts

An implementation of [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) (Danihelka et al., ICLR 2022). Numba-accelerated, hundreds of thousands of simulations per second.

Gumbel MCTS replaces PUCT's UCB-based action selection with sequential halving over Gumbel-perturbed log-priors. This produces better actions under the same simulation budget — especially at low budgets, where PUCT wastes visits on bad branches it can't prune.

<table>
<tr>
<td align="center" width="50%">
<b>PushT</b> — uniform prior, 64 sims<br>
<a href="https://github.com/olivkoch/gumbel-mcts/blob/main/demo/mp4/final_pusht_uni_sims64.mp4">
<img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/demo/gif/final_pusht_uni_sims64.gif" width="100%" alt="PushT: PUCT IoU=0.03 vs Gumbel IoU=0.85" />
</a>
</td>
<td align="center" width="50%">
<b>Wall-gap PushT</b> — push through a narrow gap, 64 sims<br>
<a href="https://github.com/olivkoch/gumbel-mcts/blob/main/demo/mp4/final_wallgap_sims64.mp4">
<img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/demo/gif/final_wallgap_sims64.gif" width="100%" alt="Wall-gap: PUCT frac=0.50 vs Gumbel frac=1.00" />
</a>
</td>
</tr>
<tr>
<td align="center">
<b>Robot arm</b> — 4 joints, 2 obstacles, 64 sims<br>
<a href="https://github.com/olivkoch/gumbel-mcts/blob/main/demo/mp4/final_robot_arm.mp4">
<img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/demo/gif/final_robot_arm.gif" width="100%" alt="Robot arm: PUCT dist=306 vs Gumbel dist=3" />
</a>
</td>
<td align="center">
<b>Sokoban</b> — 2-box puzzle, 32 sims<br>
<a href="https://github.com/olivkoch/gumbel-mcts/blob/main/demo/mp4/sokoban_anim.mp4">
<img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/demo/gif/sokoban_anim.gif" width="100%" alt="Sokoban: PUCT fails vs Gumbel solves" />
</a>
</td>
</tr>
</table>

<p align="center"><i>PUCT (left) vs Gumbel (right), same budget, same model. Click to play.</i></p>

> All demos are fully reproducible — see [`demo/`](demo/) for scripts, sweep results, and instructions.

## What's included

Three MCTS implementations sharing the same tree storage and batch interface:

| Algorithm | File | Best for |
|-----------|------|----------|
| **PUCT** | `puct.py` | Standard UCB-based MCTS. 2-20x faster than [reference](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py) on CPU and GPU. |
| **Gumbel Dense** | `gumbel_dense.py` | Low-budget planning. Sequential halving over all legal actions. |
| **Gumbel Sparse** | `gumbel_sparse.py` | Large action spaces (e.g. chess). Samples a subset of actions to consider. |

See [gumbel-mcts-benchmark](https://github.com/olivkoch/gumbel-mcts-benchmark) for validation against a gold-standard MCTS.

## Installation

```bash
pip install gumbel-mcts
```

## Quick start

```python
from gumbel_mcts import GumbelDense

tree = GumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
tree.initialize_roots([0], board[None], np.array([player]))
action = tree.run_simulation_batch(model, [0], num_simulations=64)[0]
```

The interface is the same for all three algorithms — swap `GumbelDense` for `PUCT` or `GumbelSparse`.

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/gumbel.png" width="100%" alt="Gumbel sequential halving" /><br>
  <small><i><a href="https://medium.com/correll-lab/planning-with-gumbel-036018b180bf">Improving MuZero using the Gumbel top-k trick</a>, by Xavier O'Keefe</i></small>
</p>

Standard PUCT explores actions proportionally to a UCB score that mixes prior probability and visit-based value estimates. When the budget is small, most actions get only a few visits, and the value estimates are noisy — PUCT can't reliably distinguish good actions from bad ones.

Gumbel MCTS takes a different approach: add Gumbel noise to the log-prior to create an initial ranking, then use sequential halving to eliminate half the candidates at each phase. Each phase allocates equal simulations to the surviving candidates. The result is that the budget concentrates on a shrinking set of promising actions, producing a much sharper signal with the same total simulation count.

## Board game results

Head-to-head on 15x15 Gomoku across simulation budgets. Gumbel's advantage is largest at low budgets and persists as the model improves:

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/puct_vs_gumbel_winrate_random.png" width="100%" alt="PUCT vs Gumbel — random model" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/puct_vs_gumbel_winrate_heuristic.png" width="100%" alt="PUCT vs Gumbel — heuristic model" />
</p>

With 8 simulations on 9x9 Gomoku, Gumbel concentrates visits on the strategic moves while PUCT spreads them across irrelevant squares:

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/gomoku_heatmap_9x9.png" width="100%" alt="Visit distribution heatmap — PUCT vs Gumbel Dense" />
</p>
