"""Protocols defining the interfaces MCTS expects from games and models."""

from typing import Protocol, Tuple, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class GameLogic(Protocol):
    """Interface a game must implement to work with MCTS."""

    NUM_ACTIONS: int        # Total action space size (e.g. 7 for Connect4, 4672 for chess)
    BOARD_SHAPE: tuple      # Shape of a single board state (e.g. (6,7) or (8,8))
    MAX_MOVES: int          # Maximum game length before forced draw
    MAX_LEGAL_MOVES: int    # Upper bound on legal moves in any position
    PLAYER_1: int           # Player 1 identifier (typically 1)
    PLAYER_2: int           # Player 2 identifier (typically 2)

    def fast_step(
        self, board: np.ndarray, action: int, player: int
    ) -> Tuple[float, int, bool, np.ndarray]:
        """Execute a move.

        Returns:
            reward: +1 if current player wins, -1 if loses, 0 otherwise
            winner: player id of winner, or 0 if no winner yet
            done:   whether the game is over
            new_board: the resulting board state
        """
        ...

    def get_valid_mask(
        self, board: np.ndarray, player: int
    ) -> np.ndarray:
        """Return a (NUM_ACTIONS,) float32 mask: 1.0 for legal moves, 0.0 otherwise."""
        ...


@runtime_checkable
class MCTSModel(Protocol):
    """Interface a neural network must implement to work with MCTS."""

    logic: GameLogic

    def forward_for_mcts(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run inference for MCTS.

        Args:
            batch: dict with keys:
                "boards":             (B, *BOARD_SHAPE) float tensor
                "current_player":     (B,) int tensor
                "legal_actions_mask": (B, NUM_ACTIONS) bool tensor  [optional, v5 only]

        Returns:
            dict with keys:
                "policy": (B, NUM_ACTIONS) float tensor (probabilities, sums to 1)
                "value":  (B,) or (B, 1) float tensor (in [-1, 1])
        """
        ...
