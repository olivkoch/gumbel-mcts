"""Tic-Tac-Toe game logic implementing the GameLogic protocol."""

import numpy as np
from numba import njit


@njit(cache=True)
def _board_line_match(line, p):
    for v in line:
        if v != p:
            return False
    return True


@njit(cache=True)
def fast_step(board, action, player):
    r = action // 3
    c = action % 3
    board[r, c] = player
    for p in np.array([1, 2], dtype=np.int8):
        for i in range(3):
            if _board_line_match(board[i], p) or _board_line_match(board[:, i], p):
                reward = 1.0 if p == player else -1.0
                return reward, p, True, board
        if _board_line_match(np.array([board[0, 0], board[1, 1], board[2, 2]]), p):
            reward = 1.0 if p == player else -1.0
            return reward, p, True, board
        if _board_line_match(np.array([board[0, 2], board[1, 1], board[2, 0]]), p):
            reward = 1.0 if p == player else -1.0
            return reward, p, True, board
    all_filled = True
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                all_filled = False
    if all_filled:
        return 0.0, 0, True, board
    return 0.0, 0, False, board


@njit(cache=True)
def get_valid_mask(board, player):
    mask = np.zeros(9, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                mask[i * 3 + j] = 1.0
    return mask


class TicTacToeLogic:
    GAME_NAME = 'tictactoe'
    NUM_ACTIONS = 9
    BOARD_SHAPE = (3, 3)
    MAX_MOVES = 9
    MAX_LEGAL_MOVES = 9
    PLAYER_1 = 1
    PLAYER_2 = 2
    NODE_STORAGE_WIDTH = 9
    NN_OBS_WIDTH = 9
    USE_HISTORY = False
    HISTORY_STEPS = 0

    fast_step = staticmethod(fast_step)
    get_valid_mask = staticmethod(get_valid_mask)

    @staticmethod
    def get_initial_board():
        return np.zeros((3, 3), dtype=np.int8)
