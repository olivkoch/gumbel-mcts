"""Tests for Gumbel Dense MCTS implementation."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_logic.gomoku import GomokuLogic
from gumbel_mcts.gumbel_dense import GumbelDense

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
        # Put 90% on preferred, spread rest uniformly
        eps = 0.1 / (self.logic.NUM_ACTIONS - 1)
        policy = torch.full((b, self.logic.NUM_ACTIONS), eps, dtype=torch.float32)
        policy[:, self.preferred_action] = 0.9
        value = torch.full((b, 1), self._value, dtype=torch.float32)
        return {"policy": policy, "value": value}


@pytest.fixture
def logic():
    return GomokuLogic()


@pytest.fixture
def model(logic):
    return DummyModel(logic)


@pytest.fixture
def gumbel(logic):
    return GumbelDense(n_games=1, max_nodes=2000, logic=logic, device="cpu")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestGumbelDenseInit:
    def test_inherits_puct_storage(self, gumbel):
        assert hasattr(gumbel, 'storage')
        assert gumbel.storage.children.shape[1] == NUM_ACTIONS

    def test_gumbel_specific_arrays(self, gumbel):
        assert gumbel.gumbel_noises.shape == (1, NUM_ACTIONS)
        assert gumbel.root_logits.shape == (1, NUM_ACTIONS)
        assert gumbel.root_legal_masks.shape == (1, NUM_ACTIONS)
        assert gumbel.root_nn_values.shape == (1,)

    def test_default_c_constants(self, logic):
        g = GumbelDense(n_games=1, max_nodes=100, logic=logic, device="cpu")
        assert g.c_visit == 50.0
        assert g.c_scale == 1.0

    def test_custom_c_constants(self, logic):
        g = GumbelDense(n_games=1, max_nodes=100, logic=logic, device="cpu",
                        c_visit=30.0, c_scale=2.0)
        assert g.c_visit == 30.0
        assert g.c_scale == 2.0

    def test_multi_game_shapes(self, logic):
        g = GumbelDense(n_games=4, max_nodes=2000, logic=logic, device="cpu")
        assert g.gumbel_noises.shape == (4, NUM_ACTIONS)
        assert g.root_logits.shape == (4, NUM_ACTIONS)
        assert g.root_nn_values.shape == (4,)


# ---------------------------------------------------------------------------
# initialize_roots
# ---------------------------------------------------------------------------

class TestInitializeRoots:
    def test_gumbel_noise_populated(self, gumbel, logic):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        # Gumbel noise should not be all zeros after initialization
        assert not np.allclose(gumbel.gumbel_noises[0], 0.0)

    def test_root_node_created(self, gumbel, logic):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        root_idx = gumbel.storage.root_indices[0]
        assert root_idx >= 1
        assert gumbel.storage.players[root_idx] == logic.PLAYER_1

    def test_multiple_games(self, logic):
        g = GumbelDense(n_games=3, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board() for _ in range(3)]
        players = [logic.PLAYER_1] * 3
        g.initialize_roots([0, 1, 2], boards, players)
        # Each game should get distinct Gumbel noise
        assert not np.array_equal(g.gumbel_noises[0], g.gumbel_noises[1])


# ---------------------------------------------------------------------------
# run_simulation_batch
# ---------------------------------------------------------------------------

class TestRunSimulationBatch:
    def test_returns_moves_array(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gumbel.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)
        assert 0 <= moves[0] < logic.NUM_ACTIONS

    def test_returned_move_is_legal(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gumbel.run_simulation_batch(model, [0], num_simulations=16)
        valid = logic.get_valid_mask(board, logic.PLAYER_1)
        assert valid[moves[0]] == 1.0

    def test_root_expanded_and_visited(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        root_idx = gumbel.storage.root_indices[0]
        assert gumbel.storage.is_expanded[root_idx]
        assert gumbel.storage.visit_counts[root_idx] > 1

    def test_multi_game(self, logic):
        g = GumbelDense(n_games=2, max_nodes=4000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        g.initialize_roots([0, 1], boards, players)
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0, 1], num_simulations=16)
        assert moves.shape == (2,)
        for m in moves:
            assert 0 <= m < logic.NUM_ACTIONS

    def test_children_created(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        root_idx = gumbel.storage.root_indices[0]
        assert np.any(gumbel.storage.children[root_idx] != -1)

    def test_small_budget(self, gumbel, logic, model):
        """Even with very few simulations, should not crash."""
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gumbel.run_simulation_batch(model, [0], num_simulations=4)
        assert moves.shape == (1,)


# ---------------------------------------------------------------------------
# _expand_roots_v4 (indirectly via run_simulation_batch)
# ---------------------------------------------------------------------------

class TestExpandRoots:
    def test_root_logits_populated(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        # With uniform policy, logits should be log(1/225) ≈ -5.416
        expected = np.log(1.0 / NUM_ACTIONS + 1e-10)
        np.testing.assert_allclose(gumbel.root_logits[0], expected, atol=1e-4)

    def test_root_legal_masks_populated(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        # Empty board: all 225 moves legal
        assert gumbel.root_legal_masks[0].sum() == NUM_ACTIONS

    def test_root_nn_values_populated(self, gumbel, logic):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        model = BiasedModel(logic, preferred_action=CENTER, value=0.5)
        gumbel._expand_roots_v4(model, [0])
        np.testing.assert_allclose(gumbel.root_nn_values[0], 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# _get_initial_gumbel_candidates
# ---------------------------------------------------------------------------

class TestGetInitialCandidates:
    def test_candidate_count(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        game_indices = np.array([0], dtype=np.int32)
        mask = gumbel._get_initial_gumbel_candidates(logic, game_indices, k=4)
        assert mask.shape == (1, NUM_ACTIONS)
        assert mask[0].sum() == 4

    def test_k_larger_than_legal(self, gumbel, logic, model):
        """When k > legal moves, all legal moves are candidates."""
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        game_indices = np.array([0], dtype=np.int32)
        mask = gumbel._get_initial_gumbel_candidates(logic, game_indices, k=20)
        assert mask[0].sum() == 20  # k=20 is less than 225 legal moves

    def test_only_legal_moves_selected(self, gumbel, logic, model):
        """Candidates should only be legal moves."""
        board = logic.get_initial_board()
        board[0, 0] = 1  # occupy one square
        gumbel.initialize_roots([0], [board], [logic.PLAYER_2])
        gumbel._expand_roots_v4(model, [0])
        game_indices = np.array([0], dtype=np.int32)
        mask = gumbel._get_initial_gumbel_candidates(logic, game_indices, k=4)
        # Action 0 (row=0, col=0) is occupied, should not be a candidate
        assert not mask[0, 0]


# ---------------------------------------------------------------------------
# _halve_candidates
# ---------------------------------------------------------------------------

class TestHalveCandidates:
    def test_halves_count(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        game_indices = np.array([0], dtype=np.int32)

        # Start with 4 candidates
        mask = gumbel._get_initial_gumbel_candidates(logic, game_indices, k=4)
        assert mask[0].sum() == 4

        # Run a few sims so children exist
        gumbel.run_simulation_batch(model, [0], num_simulations=16)

        # Re-create candidates after search (need to re-expand for clean state)
        gumbel2 = GumbelDense(n_games=1, max_nodes=2000, logic=logic, device="cpu")
        gumbel2.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel2._expand_roots_v4(model, [0])
        mask2 = gumbel2._get_initial_gumbel_candidates(logic, game_indices, k=4)
        halved = gumbel2._halve_candidates(game_indices, mask2)
        assert halved[0].sum() == 2

    def test_single_candidate_preserved(self, gumbel, logic, model):
        """Halving a single candidate should keep it."""
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel._expand_roots_v4(model, [0])
        game_indices = np.array([0], dtype=np.int32)
        mask = np.zeros((1, NUM_ACTIONS), dtype=np.bool_)
        mask[0, CENTER] = True  # only centre
        halved = gumbel._halve_candidates(game_indices, mask)
        assert halved[0].sum() == 1
        assert halved[0, CENTER]


# ---------------------------------------------------------------------------
# get_improved_policy
# ---------------------------------------------------------------------------

class TestGetImprovedPolicy:
    def test_shape(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        policy = gumbel.get_improved_policy(n_active=1)
        assert policy.shape == (1, NUM_ACTIONS)

    def test_sums_to_one(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        policy = gumbel.get_improved_policy(n_active=1)
        np.testing.assert_allclose(policy[0].sum(), 1.0, atol=1e-5)

    def test_non_negative(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        policy = gumbel.get_improved_policy(n_active=1)
        assert np.all(policy >= 0)

    def test_zero_on_illegal_moves(self, logic):
        """Illegal moves should have zero probability."""
        g = GumbelDense(n_games=1, max_nodes=2000, logic=logic, device="cpu")
        board = logic.get_initial_board()
        board[0, 0] = 1; board[1, 1] = 2  # occupy some squares
        g.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel(logic)
        g.run_simulation_batch(model, [0], num_simulations=16)
        policy = g.get_improved_policy(n_active=1)
        assert policy[0, 0] == 0.0  # occupied
        assert policy[0, 16] == 0.0  # occupied (action 16 = row 1, col 1)


# ---------------------------------------------------------------------------
# get_gumbel_root_value
# ---------------------------------------------------------------------------

class TestGetGumbelRootValue:
    def test_shape(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        v = gumbel.get_gumbel_root_value(n_active=1)
        assert v.shape == (1,)
        assert v.dtype == np.float32

    def test_with_chosen_moves(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gumbel.run_simulation_batch(model, [0], num_simulations=16)
        v = gumbel.get_gumbel_root_value(n_active=1, chosen_moves=moves)
        assert v.shape == (1,)
        assert -1.0 <= v[0] <= 1.0

    def test_without_chosen_moves(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        v = gumbel.get_gumbel_root_value(n_active=1)
        assert -1.0 <= v[0] <= 1.0


# ---------------------------------------------------------------------------
# get_all_root_data (overridden)
# ---------------------------------------------------------------------------

class TestGetAllRootData:
    def test_shape(self, logic):
        g = GumbelDense(n_games=2, max_nodes=2000, logic=logic, device="cpu")
        boards = [logic.get_initial_board(), logic.get_initial_board()]
        players = [logic.PLAYER_1, logic.PLAYER_1]
        g.initialize_roots([0, 1], boards, players)
        model = DummyModel(logic)
        g.run_simulation_batch(model, [0, 1], num_simulations=16)
        visits, q = g.get_all_root_data(n_active=2)
        assert visits.shape == (2, NUM_ACTIONS)
        assert q.shape == (2,)

    def test_visits_non_negative(self, gumbel, logic, model):
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        visits, _ = gumbel.get_all_root_data(n_active=1)
        assert np.all(visits >= 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_nodes_boundary(self, logic):
        """Small max_nodes should not crash."""
        g = GumbelDense(n_games=1, max_nodes=20, logic=logic, device="cpu")
        board = logic.get_initial_board()
        g.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)

    def test_reset_and_reuse(self, gumbel, logic, model):
        """Reset + re-init + re-search should work."""
        board = logic.get_initial_board()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        gumbel.run_simulation_batch(model, [0], num_simulations=16)
        gumbel.reset()
        gumbel.initialize_roots([0], [board], [logic.PLAYER_1])
        moves = gumbel.run_simulation_batch(model, [0], num_simulations=16)
        assert moves.shape == (1,)

    def test_partially_filled_board(self, logic):
        """Search on a board with some moves already played."""
        g = GumbelDense(n_games=1, max_nodes=2000, logic=logic, device="cpu")
        board = logic.get_initial_board()
        board[0, 0] = 1; board[0, 1] = 2; board[1, 0] = 1; board[1, 1] = 2
        g.initialize_roots([0], [board], [logic.PLAYER_1])
        model = DummyModel(logic)
        moves = g.run_simulation_batch(model, [0], num_simulations=16)
        # Returned move should be among the remaining empty squares
        valid = logic.get_valid_mask(board, logic.PLAYER_1)
        assert valid[moves[0]] == 1.0

    def test_biased_model_influences_move(self, logic):
        """A model biased toward centre should pick centre more often."""
        results = []
        for _ in range(5):
            g = GumbelDense(n_games=1, max_nodes=2000, logic=logic, device="cpu")
            board = logic.get_initial_board()
            g.initialize_roots([0], [board], [logic.PLAYER_1])
            model = BiasedModel(logic, preferred_action=CENTER, value=0.0)
            moves = g.run_simulation_batch(model, [0], num_simulations=32)
            results.append(moves[0])
        # Centre (action 112) should appear at least once in 5 trials
        assert CENTER in results
