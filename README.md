[![PyPI version](https://badge.fury.io/py/gumbel-mcts.svg)](https://pypi.org/project/gumbel-mcts/)
[![Tests](https://github.com/olivkoch/gumbel-mcts/actions/workflows/tests.yml/badge.svg)](https://github.com/olivkoch/gumbel-mcts/actions)
[![codecov](https://codecov.io/gh/olivkoch/gumbel-mcts/branch/main/graph/badge.svg)](https://codecov.io/gh/olivkoch/gumbel-mcts)
[![docs](https://readthedocs.org/projects/gumbel-mcts/badge/?version=latest)](https://gumbel-mcts.readthedocs.io)
[![PyPI Downloads](https://img.shields.io/pypi/dm/gumbel-mcts)](https://pypi.org/project/gumbel-mcts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# gumbel-mcts

A lightweight and numba-accelerated Gumbel MCTS implementation. 

Optimized for speed! Generates hundreds of thousands of sims / sec. :rocket:

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/gumbel.png" width="100%" alt="Gumbel principle" /><br>
  <small><i><a href="https://medium.com/correll-lab/planning-with-gumbel-036018b180bf">Improving MuZero using the Gumbel top-k trick</a>, by Xavier O'Keefe</i></small>
</p>

## Description

Gumbel sampling brought tremendous progress to MCTS, but efficient standalone implementation of Gumbel MCTS are missing.

We provide three MCTS implementations:

- `puct.py`: an efficient implementation of PUCT MCTS. It produces the exact same output as a reference [mcts_v2.py](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py) but but with a **2-20X speedup** on both Mac and NVIDIA GPUs. 

- `gumbel_dense.py`: an implementation of [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO), offering **massive learning efficiency when the simulation budget is low**

- `gumbel_sparse.py`: a sparse implementation of Gumbel MCTS, particularly useful for games with large action spaces (e.g. chess)

Our Gumbel implementation offers **both simulation efficiency and speed**.

See [gumbel-mcts-benchmark](https://github.com/olivkoch/gumbel-mcts-benchmark) for full benchmark and validation against a gold standard MCTS.

## Usage

```python

def play_game():
    logic = TicTacToeLogic()
    model = TinyModel()
    model.eval()

    board = np.zeros((3, 3), dtype=np.int8)
    player = 1
    symbols = {0: ".", 1: "X", 2: "O"}

    while True:
        tree = GumbelSparse(n_games=1, max_nodes=500, device="cpu", logic=logic)
        tree.initialize_roots([0], board.ravel()[None], np.array([player]))
        move = tree.run_simulation_batch(model, [0], num_simulations=50)
        action = move[0]

        _, winner, done, board = logic.fast_step(board, action, player)
```

## Illustration

With a random model, Gumbel wins with low-budget but PUCT catches up. As soon as the model gets better than random, Gumbel wins.

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/puct_vs_gumbel_winrate_random.png" width="100%" alt="PUCT vs Gumbel" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/puct_vs_gumbel_winrate_heuristic.png" width="100%" alt="PUCT vs Gumbel" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/puct_vs_gumbel_winrate_strong.png" width="100%" alt="PUCT vs Gumbel" />
</p>

Gumbel MCTS makes much better use of its simulation budget. With 8 sims on Gomoku, Gumbel finds the strategic moves while PUCT concentrates its visit counts at the wrong place.

<p align="center">
  <img src="https://raw.githubusercontent.com/olivkoch/gumbel-mcts/main/img/gomoku_heatmap_9x9.png" width="100%" alt="Gomoku Heatmap 9x9 — PUCT vs Gumbel Dense" />
</p>
