# gumbel-mcts

A lightweight and modular Gumbel MCTS implementation

## Description

Most open-source MCTS implementations provide only standard PUCT (AlphaZero-style) or dense Gumbel MCTS (MuZero, EfficientZero). Sparse Gumbel MCTS — which makes Gumbel planning practical for large action spaces like chess (4672 actions) or Go (362) — is essentially absent.

We provide three MCTS implementations:

- `puct.py`: an efficient implementation of PUCT MCTS. It produces the exact same output as a reference [mcts_v2.py](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py) but with 2-20X speedup. 

- `gumbel_dense.py`: an implementation of [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO), offering up to 200X improvement in win rate / simulation budget over PUCT.

- `gumbel_sparse.py`: a sparse implementation of Gumbel MCTS, particularly useful for games with large action spaces (e.g. chess)

See [gumbel-mcts-benchmark](https://github.com/olivkoch/gumbel-mcts-benchmark) for all benchmarks.

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

Gumbel MCTS makes much better use of its simulation budget. With 8 sims on Gomoku, Gumbel finds the strategic moves while PUCT concentrates its visit counts at the wrong place.

<p align="center">
  <img src="examples/gomoku_heatmap_9x9.png" width="700" alt="Gomoku Heatmap 9x9 — PUCT vs Gumbel Dense" />
</p>
