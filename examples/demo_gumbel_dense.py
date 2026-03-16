"""demo_gumbel_dense.py — Gumbel MCTS (dense) playing Tic-Tac-Toe against itself."""

import numpy as np
from tictactoe import TicTacToeLogic, TinyModel, print_board
from gumbel_mcts import GumbelDense


def play_game():
    logic = TicTacToeLogic()
    model = TinyModel()
    model.eval()

    board = np.zeros((3, 3), dtype=np.int8)
    player = 1

    while True:
        tree = GumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([player]))
        move = tree.run_simulation_batch(model, [0], num_simulations=50)
        action = move[0]

        _, winner, done, board = logic.fast_step(board, action, player)
        print_board(board)

        if done:
            print(f"Winner: {({0: 'Draw', 1: 'X', 2: 'O'}).get(winner, 'Draw')}")
            break
        player = 3 - player


if __name__ == "__main__":
    play_game()
