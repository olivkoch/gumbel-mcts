"""Tic-Tac-Toe helpers for the MCTS demos."""

import numpy as np
import torch
import torch.nn as nn
from game_logic.tictactoe import TicTacToeLogic  # noqa: F401


# ---------------------------------------------------------------------------
# Tiny neural network (implements the MCTSModel protocol)
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.policy_head = nn.Linear(64, 9)
        self.value_head = nn.Linear(64, 1)
        self.logic = TicTacToeLogic()

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float()
        h = self.net(boards)
        policy = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        return {"policy": policy, "value": value}


SYMBOLS = {0: ".", 1: "X", 2: "O"}


def print_board(board):
    print("\n".join(" ".join(SYMBOLS[int(c)] for c in row) for row in board))
    print()
