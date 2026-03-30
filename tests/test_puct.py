"""Tests for PUCT MCTS implementation."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_logic.gomoku import GomokuLogic
from gumbel_mcts.puct import PUCT, PUCTStorage

NUM_ACTIONS = 225
CENTER = 112  # 7*15+7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyModel:
    """A mock model returning uniform policy and zero value."""

    def __init__(self, num_actions=NUM_ACTIONS):
        self.num_actions = num_actions

    def forward_for_mcts(self, batch):
        b = batch["boards"].shape[0]
        policy = torch.ones(b, self.num_actions, dtype=torch.float32) / self.num_actions
        value = torch.zeros(b, 1, dtype=torch.float32)
        return {"policy": policy, "value": value}


class BiasedModel:
    """A model that puts all probability on a specified action."""

    def __init__(self, preferred_action, num_actions=NUM_ACTIONS, value=0.0):
        self.preferred_action = preferred_action
        self.num_actions = num_actions
        self._value = value

    def forward_for_mcts(self, batch):
        b = batch["boards"].shape[0]
        policy = torch.zeros(b, self.num_actions, dtype=torch.float32)
        policy[:, self.preferred_action] = 1.0
        value = torch.full((b, 1), self._value, dtype=torch.float32)
        return {"policy": policy, "value": value}


@pytest.fixture
def logic():
    return GomokuLogic()


@pytest.fixture
def puct(logic):
    return PUCT(n_games=1, max_nodes=2000, logic=logic, device="cpu")


# ---------------------------------------------------------------------------
# PUCTStorage tests
# ---------------------------------------------------------------------------

class TestPUCTStorage:
    def test_init_shapes(self, logic):
        s = PUCTStorage(n_games=2, max_nodes=64, logic=logic)
        assert s.children.shape == (64, NUM_ACTIONS)
        assert s.parents.shape == (64,)
        assert s.visit_counts.shape == (64,)
        assert s.values.shape == (64,)
        assert s.prior_probs.shape == (64, NUM_ACTIONS)
        assert s.boards.shape == (64, 15, 15)
        assert s.players.shape == (64,)
        assert s.root_indices.shape == (2,)

    def test_init_defaults(self, logic):
        s = PUCTStorage(n_games=1, max_nodes=32, logic=logic)
        assert np.all(s.children == -1)
        assert np.all(s.parents == -1)
        assert np.all(s.visit_counts == 0)
        assert np.all(s.values == 0.0)


# ---------------------------------------------------------------------------
# PUCT constructor tests
# ---------------------------------------------------------------------------

class TestPUCTInit:
    def test_cpu_device(self, logic):
        p = PUCT(n_games=1, max_nodes=100, logic=logic, device="cpu")
        assert p.device == "cpu"
        assert p._use_autocast is False

    def test_cuda_device(self, logic):
        p = PUCT(n_games=1, max_nodes=100, logic=logic, device="cuda")
        assert p._use_autocast is True
        assert p._autocast_device == "cuda"

    def test_mps_device(self, logic):
        p = PUCT(n_games=1, max_nodes=100, logic=logic, device="mps")
        assert p._use_autocast is False
        assert p._autocast_device == "mps"

    def test_next_free_idx_starts_at_one(self, puct):
        assert puct.next_free_idx_arr[0] == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_index(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        assert puct.next_free_idx_arr[0] > 1
        puct.reset()
        assert puct.next_free_idx_arr[0] == 1

    def test_reset_clears_flags(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=5)
        puct.reset()
        # All nodes in the used range should be cleared
        assert not np.any(puct.storage.is_expanded[:2])


# ---------------------------------------------------------------------------
# Initialize roots
# ---------------------------------------------------------------------------

class TestInitializeRoots:
    def test_single_game(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        root_idx = puct.storage.root_indices[0]
        assert root_idx == 1  # index 0 is unused sentinel
        assert puct.storage.players[root_idx] == logic.PLAYER_1
        assert np.array_equal(puct.storage.boards[root_idx], board)
        assert puct.storage.visit_counts[root_idx] == 0
        assert not puct.storage.is_expanded[root_idx]
        assert not puct.storage.is_terminal[root_idx]

    def test_multiple_games(self, logic):
        p = PUCT(n_games=3, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board() for _ in range(3)]
        players = [logic.PLAYER_1, logic.PLAYER_1, logic.PLAYER_1]
        p.initialize_roots([0, 1, 2], boards, players)
        for g in range(3):
            root_idx = p.storage.root_indices[g]
            assert root_idx >= 1
            assert p.storage.players[root_idx] == logic.PLAYER_1


# ---------------------------------------------------------------------------
# run_simulation_batch
# ---------------------------------------------------------------------------

class TestRunSimulationBatch:
    def test_root_visits_match_num_simulations(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=20)
        root_idx = puct.storage.root_indices[0]
        # Root visits should be >= num_simulations + 1 (root expansion counts)
        assert puct.storage.visit_counts[root_idx] >= 21

    def test_children_created(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=20)
        root_idx = puct.storage.root_indices[0]
        # With uniform prior and 20 sims on empty board, some children should exist
        children = puct.storage.children[root_idx]
        assert np.any(children != -1)

    def test_root_expanded_after_simulation(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=5)
        root_idx = puct.storage.root_indices[0]
        assert puct.storage.is_expanded[root_idx]

    def test_multiple_games_simulation(self, logic):
        p = PUCT(n_games=2, max_nodes=4000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        p.initialize_roots([0, 1], boards, players)
        model = DummyModel()
        p.run_simulation_batch(model, [0, 1], num_simulations=10)
        for g in [0, 1]:
            root_idx = p.storage.root_indices[g]
            assert p.storage.visit_counts[root_idx] >= 11

    def test_biased_model_concentrates_visits(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = BiasedModel(preferred_action=CENTER)  # centre of 15x15
        puct.run_simulation_batch(model, [0], num_simulations=30)
        root_idx = puct.storage.root_indices[0]
        child_centre = puct.storage.children[root_idx, CENTER]
        assert child_centre != -1
        # Centre child should have the most visits
        centre_visits = puct.storage.visit_counts[child_centre]
        for move in range(NUM_ACTIONS):
            c = puct.storage.children[root_idx, move]
            if c != -1 and move != CENTER:
                assert centre_visits >= puct.storage.visit_counts[c]


# ---------------------------------------------------------------------------
# get_root_data
# ---------------------------------------------------------------------------

class TestGetRootData:
    def test_returns_correct_shape(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=10)
        visits, q = puct.get_root_data(0)
        assert visits.shape == (NUM_ACTIONS,)
        assert isinstance(q, (float, np.floating))

    def test_visit_sum_matches_root(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=15)
        visits, _ = puct.get_root_data(0)
        root_idx = puct.storage.root_indices[0]
        # Sum of child visits + 1 (root expansion) == root visit count
        assert int(visits.sum()) + 1 == puct.storage.visit_counts[root_idx]

    def test_no_visits_returns_zero_q(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        _, q = puct.get_root_data(0)
        assert q == 0.0


# ---------------------------------------------------------------------------
# get_all_root_data
# ---------------------------------------------------------------------------

class TestGetAllRootData:
    def test_shape(self, logic):
        p = PUCT(n_games=3, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board() for _ in range(3)]
        players = [logic.PLAYER_1] * 3
        p.initialize_roots([0, 1, 2], boards, players)
        model = DummyModel()
        p.run_simulation_batch(model, [0, 1, 2], num_simulations=5)
        visits, q = p.get_all_root_data(n_active=3)
        assert visits.shape == (3, NUM_ACTIONS)
        assert q.shape == (3,)

    def test_consistent_with_get_root_data(self, logic):
        p = PUCT(n_games=2, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        p.initialize_roots([0, 1], boards, players)
        model = DummyModel()
        p.run_simulation_batch(model, [0, 1], num_simulations=10)
        all_visits, all_q = p.get_all_root_data(n_active=2)
        for g in range(2):
            single_visits, single_q = p.get_root_data(g)
            np.testing.assert_array_equal(all_visits[g], single_visits)
            np.testing.assert_allclose(all_q[g], single_q, atol=1e-6)


# ---------------------------------------------------------------------------
# get_max_depth
# ---------------------------------------------------------------------------

class TestGetMaxDepth:
    def test_zero_before_simulation(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        assert puct.get_max_depth() == 0

    def test_positive_after_simulation(self, puct, logic):
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=20)
        assert puct.get_max_depth() > 0

    def test_empty_tree(self, puct):
        assert puct.get_max_depth() == 0


# ---------------------------------------------------------------------------
# get_all_root_visits
# ---------------------------------------------------------------------------

class TestGetAllRootVisits:
    def test_returns_visits_only(self, logic):
        p = PUCT(n_games=2, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        p.initialize_roots([0, 1], boards, players)
        model = DummyModel()
        p.run_simulation_batch(model, [0, 1], num_simulations=5)
        visits = p.get_all_root_visits(n_active=2)
        assert visits.shape == (2, NUM_ACTIONS)

    def test_default_n_active(self, logic):
        p = PUCT(n_games=2, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        p.initialize_roots([0, 1], boards, players)
        model = DummyModel()
        p.run_simulation_batch(model, [0, 1], num_simulations=5)
        visits = p.get_all_root_visits()
        assert visits.shape == (2, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_terminal_root_no_crash(self, logic):
        """Initialize root with a board that's already won, then simulate."""
        p = PUCT(n_games=1, max_nodes=500, logic=logic, device="cpu")
        board = logic.get_initial_board()
        # Player 1 wins with 5 in a row horizontally
        for c in range(5):
            board[0, c] = 1
        p.initialize_roots([0], [board], [logic.PLAYER_2])
        model = DummyModel()
        # Should not crash — root is already terminal-like
        p.run_simulation_batch(model, [0], num_simulations=5)
        visits, q = p.get_root_data(0)
        assert visits.shape == (NUM_ACTIONS,)

    def test_max_nodes_boundary(self, logic):
        """Small max_nodes — should not crash, just stop expanding."""
        p = PUCT(n_games=1, max_nodes=20, logic=logic, device="cpu")
        board = logic.get_initial_board()
        p.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        p.run_simulation_batch(model, [0], num_simulations=50)
        # Should complete without error
        root_idx = p.storage.root_indices[0]
        assert p.storage.visit_counts[root_idx] >= 1

    def test_reinitialize_after_reset(self, puct, logic):
        """Reset and re-initialize should work cleanly."""
        board = logic.get_initial_board()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel()
        puct.run_simulation_batch(model, [0], num_simulations=10)
        puct.reset()
        puct.initialize_roots([0], [board], [logic.PLAYER_1])
        puct.run_simulation_batch(model, [0], num_simulations=10)
        root_idx = puct.storage.root_indices[0]
        assert puct.storage.visit_counts[root_idx] >= 11
