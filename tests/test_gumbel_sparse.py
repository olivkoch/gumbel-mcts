"""Tests for Gumbel Sparse MCTS implementation."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_logic.gomoku import GomokuLogic
from gumbel_mcts.gumbel_sparse import GumbelSparse, GumbelSparseStorage

NUM_ACTIONS = 225
CENTER = 112  # 7*15+7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyModel:
    """Mock model returning uniform policy and zero value."""

    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        b = batch["boards"].shape[0]
        policy = torch.ones(b, self.logic.NUM_ACTIONS, dtype=torch.float32) / self.logic.NUM_ACTIONS
        value = torch.zeros(b, 1, dtype=torch.float32)
        return {"policy": policy, "value": value}


class BiasedModel:
    """Model that concentrates probability on a single action."""

    def __init__(self, logic, preferred_action, value=0.0):
        self.logic = logic
        self.preferred_action = preferred_action
        self._value = value

    def forward_for_mcts(self, batch):
        b = batch["boards"].shape[0]
        eps = 0.1 / (self.logic.NUM_ACTIONS - 1)
        policy = torch.full((b, self.logic.NUM_ACTIONS), eps, dtype=torch.float32)
        policy[:, self.preferred_action] = 0.9
        value = torch.full((b, 1), self._value, dtype=torch.float32)
        return {"policy": policy, "value": value}


def _flat_board(logic, board_2d):
    """Flatten a 2D board to the NODE_STORAGE_WIDTH expected by sparse storage."""
    return board_2d.ravel()[:logic.NODE_STORAGE_WIDTH]


@pytest.fixture
def logic():
    return GomokuLogic()


@pytest.fixture
def model(logic):
    return DummyModel(logic)


@pytest.fixture
def gs(logic):
    return GumbelSparse(
        n_games=1, max_nodes=2000, logic=logic, device="cpu",
        avg_branching=35, max_legal_moves=NUM_ACTIONS,
    )


# ---------------------------------------------------------------------------
# GumbelSparseStorage tests
# ---------------------------------------------------------------------------

class TestGumbelSparseStorage:
    def test_init_shapes(self, logic):
        s = GumbelSparseStorage(
            max_nodes=64, max_edges=200, n_games=2,
            board_shape=logic.BOARD_SHAPE, num_actions=logic.NUM_ACTIONS, logic=logic,
        )
        assert s.parents.shape == (64,)
        assert s.visit_counts.shape == (64,)
        assert s.values.shape == (64,)
        assert s.boards.shape == (64, logic.NODE_STORAGE_WIDTH)
        assert s.players.shape == (64,)
        assert s.root_indices.shape == (2,)
        # Edge arrays
        assert s.edge_action.shape == (200,)
        assert s.edge_child.shape == (200,)
        assert s.edge_prior.shape == (200,)
        # Per-node edge metadata
        assert s.node_edge_offset.shape == (64,)
        assert s.node_num_edges.shape == (64,)

    def test_defaults(self, logic):
        s = GumbelSparseStorage(
            max_nodes=32, max_edges=100, n_games=1,
            board_shape=logic.BOARD_SHAPE, num_actions=logic.NUM_ACTIONS, logic=logic,
        )
        assert np.all(s.parents == -1)
        assert np.all(s.visit_counts == 0)
        assert np.all(s.edge_child == -1)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestGumbelSparseInit:
    def test_cpu_device(self, logic):
        g = GumbelSparse(n_games=1, max_nodes=100, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        assert g.device == "cpu"
        assert g._use_autocast is False

    def test_cuda_device(self, logic):
        g = GumbelSparse(n_games=1, max_nodes=100, logic=logic, device="cuda",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        assert g._use_autocast is True

    def test_default_c_constants(self, gs):
        assert gs.c_visit == 50.0
        assert gs.c_scale == 1.0

    def test_custom_c_constants(self, logic):
        g = GumbelSparse(n_games=1, max_nodes=100, logic=logic, device="cpu",
                         c_visit=30.0, c_scale=2.0, avg_branching=35, max_legal_moves=NUM_ACTIONS)
        assert g.c_visit == 30.0
        assert g.c_scale == 2.0

    def test_root_gumbel_arrays(self, gs):
        assert gs.root_logits.shape == (1, NUM_ACTIONS)
        assert gs.root_gumbel_noise.shape == (1, NUM_ACTIONS)
        assert gs.root_actions.shape == (1, NUM_ACTIONS)
        assert gs.root_num_legal.shape == (1,)
        assert gs.candidate_mask.shape == (1, NUM_ACTIONS)

    def test_multi_game_shapes(self, logic):
        g = GumbelSparse(n_games=4, max_nodes=200, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        assert g.root_logits.shape == (4, NUM_ACTIONS)
        assert g.root_gumbel_noise.shape == (4, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# initialize_roots
# ---------------------------------------------------------------------------

class TestInitializeRoots:
    def test_single_game(self, gs, logic):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        root_idx = gs.storage.root_indices[0]
        assert root_idx >= 1
        assert gs.storage.players[root_idx] == logic.PLAYER_1
        assert not gs.storage.is_expanded[root_idx]

    def test_multiple_games(self, logic):
        g = GumbelSparse(n_games=3, max_nodes=2000, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        boards = [_flat_board(logic, logic.get_initial_board()) for _ in range(3)]
        players = [logic.PLAYER_1] * 3
        g.initialize_roots([0, 1, 2], boards, players)
        for gidx in range(3):
            root_idx = g.storage.root_indices[gidx]
            assert root_idx >= 1
            assert g.storage.players[root_idx] == logic.PLAYER_1


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_indices(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        assert gs.next_free_node_arr[0] > 1
        gs.reset()
        assert gs.next_free_node_arr[0] == 1
        assert gs.next_free_edge_arr[0] == 0

    def test_reset_clears_flags(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=8)
        gs.reset()
        assert not np.any(gs.storage.is_expanded[:2])


# ---------------------------------------------------------------------------
# run_simulation_batch
# ---------------------------------------------------------------------------

class TestRunSimulationBatch:
    def test_returns_moves_array(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)
        assert 0 <= moves[0] < logic.NUM_ACTIONS

    def test_returned_move_is_legal(self, gs, logic, model):
        board = logic.get_initial_board()
        flat = _flat_board(logic, board)
        gs.initialize_roots([0], [flat], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(model, [0], num_simulations=16)
        valid = logic.get_valid_mask(board, logic.PLAYER_1)
        assert valid[moves[0]] == 1.0

    def test_root_expanded_and_visited(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        root_idx = gs.storage.root_indices[0]
        assert gs.storage.is_expanded[root_idx]
        assert gs.storage.visit_counts[root_idx] > 1

    def test_multi_game(self, logic):
        g = GumbelSparse(n_games=2, max_nodes=4000, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        boards = [_flat_board(logic, logic.get_initial_board()) for _ in range(2)]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        g.initialize_roots([0, 1], boards, players)
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0, 1], num_simulations=16)
        assert moves.shape == (2,)
        for m in moves:
            assert 0 <= m < logic.NUM_ACTIONS

    def test_edges_allocated(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        root_idx = gs.storage.root_indices[0]
        assert gs.storage.node_num_edges[root_idx] > 0

    def test_small_budget(self, gs, logic, model):
        """Even very few simulations should not crash."""
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(model, [0], num_simulations=4)
        assert moves.shape == (1,)

    def test_c_scale_overrides(self, gs, logic, model):
        """Passing per-game c_scale overrides should work."""
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(
            model, [0], num_simulations=16,
            c_scale_overrides=np.array([2.0], dtype=np.float32),
        )
        assert moves.shape == (1,)


# ---------------------------------------------------------------------------
# _allocate_edges
# ---------------------------------------------------------------------------

class TestAllocateEdges:
    def test_edge_allocation(self, gs, logic):
        """Manually allocate edges and verify bookkeeping."""
        node_idx = 1
        legal_moves = np.array([0, 4, 8], dtype=np.int16)
        probs = np.ones(logic.NUM_ACTIONS, dtype=np.float64) / logic.NUM_ACTIONS
        gs._allocate_edges(node_idx, legal_moves, probs)
        s = gs.storage
        assert s.node_num_edges[node_idx] == 3
        start = s.node_edge_offset[node_idx]
        np.testing.assert_array_equal(s.edge_action[start:start + 3], [0, 4, 8])
        assert np.all(s.edge_child[start:start + 3] == -1)
        # Priors should be renormalized to sum to 1
        np.testing.assert_allclose(s.edge_prior[start:start + 3].sum(), 1.0, atol=1e-6)

    def test_edge_pool_exhaustion(self, logic):
        """Should raise when edge pool is full."""
        g = GumbelSparse(n_games=1, max_nodes=10, logic=logic, device="cpu",
                         avg_branching=1, max_legal_moves=NUM_ACTIONS)  # very small edge pool
        legal_moves = np.arange(NUM_ACTIONS, dtype=np.int16)
        probs = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        # First allocation fills most of the pool
        g._allocate_edges(1, legal_moves[:10], probs)
        # Second should fail (not enough slots left)
        with pytest.raises(RuntimeError, match="Edge pool exhausted"):
            g._allocate_edges(2, legal_moves[:10], probs)


# ---------------------------------------------------------------------------
# _init_candidates
# ---------------------------------------------------------------------------

class TestInitCandidates:
    def test_candidate_count(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs._expand_roots(model, [0])
        gs._init_candidates(n_active=1, k=4)
        assert gs.candidate_mask[0].sum() == 4

    def test_k_exceeds_legal(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs._expand_roots(model, [0])
        gs._init_candidates(n_active=1, k=20)
        assert gs.candidate_mask[0].sum() == 20  # k=20 < 225 legal moves


# ---------------------------------------------------------------------------
# _halve_candidates
# ---------------------------------------------------------------------------

class TestHalveCandidates:
    def test_halves_count(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs._expand_roots(model, [0])
        gs._init_candidates(n_active=1, k=4)
        assert gs.candidate_mask[0].sum() == 4
        scale = np.array([gs.c_scale], dtype=np.float32)
        gs._halve_candidates(n_active=1, scale_vector=scale)
        assert gs.candidate_mask[0].sum() == 2

    def test_single_candidate_preserved(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs._expand_roots(model, [0])
        gs.candidate_mask[0] = False
        gs.candidate_mask[0, 0] = True
        scale = np.array([gs.c_scale], dtype=np.float32)
        gs._halve_candidates(n_active=1, scale_vector=scale)
        assert gs.candidate_mask[0].sum() == 1
        assert gs.candidate_mask[0, 0]


# ---------------------------------------------------------------------------
# get_improved_policy
# ---------------------------------------------------------------------------

class TestGetImprovedPolicy:
    def test_shape(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        policy = gs.get_improved_policy(n_active=1)
        assert policy.shape == (1, logic.NUM_ACTIONS)

    def test_sums_to_one(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        policy = gs.get_improved_policy(n_active=1)
        np.testing.assert_allclose(policy[0].sum(), 1.0, atol=1e-5)

    def test_non_negative(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        policy = gs.get_improved_policy(n_active=1)
        assert np.all(policy >= 0)

    def test_zero_on_illegal_moves(self, logic):
        """Illegal moves should have zero probability."""
        g = GumbelSparse(n_games=1, max_nodes=2000, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        board = logic.get_initial_board()
        board[0, 0] = 1; board[1, 1] = 2
        flat = _flat_board(logic, board)
        g.initialize_roots([0], [flat], [logic.PLAYER_1])
        model = DummyModel(logic)
        g.run_simulation_batch(model, [0], num_simulations=16)
        policy = g.get_improved_policy(n_active=1)
        assert policy[0, 0] == 0.0  # occupied
        assert policy[0, 16] == 0.0  # occupied (action 16 = row 1, col 1)


# ---------------------------------------------------------------------------
# get_gumbel_root_value
# ---------------------------------------------------------------------------

class TestGetGumbelRootValue:
    def test_shape(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        v = gs.get_gumbel_root_value(n_active=1)
        assert v.shape == (1,)
        assert v.dtype == np.float32

    def test_with_chosen_moves(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(model, [0], num_simulations=16)
        v = gs.get_gumbel_root_value(n_active=1, chosen_moves=moves)
        assert v.shape == (1,)
        assert -1.0 <= v[0] <= 1.0

    def test_without_chosen_moves(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        v = gs.get_gumbel_root_value(n_active=1)
        assert -1.0 <= v[0] <= 1.0


# ---------------------------------------------------------------------------
# get_all_root_data
# ---------------------------------------------------------------------------

class TestGetAllRootData:
    def test_shape(self, logic):
        g = GumbelSparse(n_games=2, max_nodes=2000, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        boards = [_flat_board(logic, logic.get_initial_board()) for _ in range(2)]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        g.initialize_roots([0, 1], boards, players)
        model = DummyModel(logic)
        g.run_simulation_batch(model, [0, 1], num_simulations=16)
        visits, q = g.get_all_root_data(n_active=2)
        assert visits.shape == (2, logic.NUM_ACTIONS)
        assert q.shape == (2,)

    def test_visits_non_negative(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        visits, _ = gs.get_all_root_data(n_active=1)
        assert np.all(visits >= 0)


# ---------------------------------------------------------------------------
# get_max_depth
# ---------------------------------------------------------------------------

class TestGetMaxDepth:
    def test_zero_before_simulation(self, gs, logic):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        assert gs.get_max_depth() == 0

    def test_positive_after_simulation(self, gs, logic, model):
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        assert gs.get_max_depth() > 0

    def test_empty_tree(self, gs):
        assert gs.get_max_depth() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_nodes_boundary(self, logic):
        """Small max_nodes should not crash."""
        g = GumbelSparse(n_games=1, max_nodes=20, logic=logic, device="cpu",
                         avg_branching=225, max_legal_moves=NUM_ACTIONS)
        board = _flat_board(logic, logic.get_initial_board())
        g.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)

    def test_reset_and_reuse(self, gs, logic, model):
        """Reset + reinit + re-search should work cleanly."""
        board = _flat_board(logic, logic.get_initial_board())
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        gs.run_simulation_batch(model, [0], num_simulations=16)
        gs.reset()
        gs.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gs.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)

    def test_partially_filled_board(self, logic):
        """Search on a board with moves already played."""
        g = GumbelSparse(n_games=1, max_nodes=2000, logic=logic, device="cpu",
                         avg_branching=35, max_legal_moves=NUM_ACTIONS)
        board = logic.get_initial_board()
        board[0, 0] = 1; board[0, 1] = 2; board[1, 0] = 1; board[1, 1] = 2
        flat = _flat_board(logic, board)
        g.initialize_roots([0], [flat], [logic.PLAYER_1])
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0], num_simulations=16)
        valid = logic.get_valid_mask(board, logic.PLAYER_1)
        assert valid[moves[0]] == 1.0

    def test_biased_model_influences_move(self, logic):
        """A biased model should push the search toward its preferred action."""
        results = []
        for _ in range(5):
            g = GumbelSparse(n_games=1, max_nodes=2000, logic=logic, device="cpu",
                             avg_branching=35, max_legal_moves=NUM_ACTIONS)
            board = _flat_board(logic, logic.get_initial_board())
            g.initialize_roots([0], [board], [logic.PLAYER_1])
            model = BiasedModel(logic, preferred_action=CENTER, value=0.0)
            moves = g.run_simulation_batch(model, [0], num_simulations=32)
            results.append(moves[0])
        assert CENTER in results
