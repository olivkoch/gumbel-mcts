"""Gomoku (Five in a Row) on a 15x15 board implementing the GameLogic protocol."""

import numpy as np
from numba import njit

BOARD_SIZE = 15
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE  # 225


@njit(cache=True)
def _count_direction(board, r, c, dr, dc, player):
    """Count consecutive stones of `player` starting from (r,c) in direction (dr,dc)."""
    count = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == player:
        count += 1
        rr += dr
        cc += dc
    return count


@njit(cache=True)
def _check_win(board, r, c, player):
    """Check if placing at (r,c) creates 5-in-a-row for player."""
    # Four directions: horizontal, vertical, diagonal, anti-diagonal
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        total = 1 + _count_direction(board, r, c, dr, dc, player) + \
                    _count_direction(board, r, c, -dr, -dc, player)
        if total >= 5:
            return True
    return False


@njit(cache=True)
def fast_step(board, action, player):
    r = action // BOARD_SIZE
    c = action % BOARD_SIZE
    board[r, c] = player

    if _check_win(board, r, c, player):
        return 1.0, player, True, board

    # Check draw (board full)
    all_filled = True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i, j] == 0:
                all_filled = False
                break
        if not all_filled:
            break
    if all_filled:
        return 0.0, 0, True, board

    return 0.0, 0, False, board


@njit(cache=True)
def get_valid_mask(board, player):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i, j] == 0:
                mask[i * BOARD_SIZE + j] = 1.0
    return mask


class GomokuLogic:
    GAME_NAME = 'gomoku'
    NUM_ACTIONS = NUM_ACTIONS        # 225
    BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)  # (15, 15)
    MAX_MOVES = NUM_ACTIONS          # 225
    MAX_LEGAL_MOVES = NUM_ACTIONS    # 225
    PLAYER_1 = 1
    PLAYER_2 = 2
    NODE_STORAGE_WIDTH = NUM_ACTIONS  # 225
    NN_OBS_WIDTH = NUM_ACTIONS        # 225
    USE_HISTORY = False
    HISTORY_STEPS = 0

    fast_step = staticmethod(fast_step)
    get_valid_mask = staticmethod(get_valid_mask)

    @staticmethod
    def get_initial_board():
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
