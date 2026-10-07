"""Tests for pure-Python MCTS kernels (python_kernels.py).

These kernels are the non-numba equivalents used for single-player
environments. Testing them directly improves coverage since coverage.py
can't instrument numba-compiled code.
"""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

from gumbel_mcts.puct import PUCT
from gumbel_mcts.gumbel_dense import GumbelDense
from kernels.python_kernels import (
    select_leaves_batch,
    backpropagate_batch,
    descend_tree_kernel,
    get_gumbel_score_kernel,
    get_forced_root_moves_kernel,
    compute_gumbel_policy_kernel,
)


# ---------------------------------------------------------------------------
# Minimal single-player game: reach zero from N by subtracting 1, 2, or 3
# ---------------------------------------------------------------------------

NUM_ACTIONS = 4  # subtract 1, 2, 3, or no-op

def _countdown_step(board, action, player):
    action = int(action)
    if action == 3:  # no-op
        return 0.0, 0, False, board
    subtract = action + 1
    board = board.copy()
    val = board[0]
    new_val = max(0, val - subtract)
    board[0] = new_val
    done = new_val == 0
    reward = 1.0 if done else 0.0
    return reward, int(done), done, board


def _countdown_valid(board, player):
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    if board[0] <= 0:
        mask[:3] = 0.0
    return mask


class CountdownLogic:
    NUM_ACTIONS = NUM_ACTIONS
    BOARD_SHAPE = (1,)
    BOARD_DTYPE = np.float64
    MAX_MOVES = 50
    MAX_LEGAL_MOVES = NUM_ACTIONS
    PLAYER_1 = 1
    PLAYER_2 = 1
    fast_step = staticmethod(_countdown_step)
    get_valid_mask = staticmethod(_countdown_valid)

    def get_initial_board(self):
        return np.array([5.0], dtype=np.float64)


class CountdownModel:
    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float().cpu()
        policy = torch.ones(B, NUM_ACTIONS) / NUM_ACTIONS
        value = torch.zeros(B, 1)
        for b in range(B):
            v = boards[b, 0].item()
            value[b] = 1.0 if v <= 0 else max(0.0, 1.0 - v / 10.0)
        return {"policy": policy, "value": value}


class BiasedCountdownModel:
    """Puts 90% probability on subtract-3 (action 2)."""
    def __init__(self, logic):
        self.logic = logic

    def forward_for_mcts(self, batch):
        B = batch["boards"].shape[0]
        boards = batch["boards"].float().cpu()
        policy = torch.full((B, NUM_ACTIONS), 0.033)
        policy[:, 2] = 0.9
        value = torch.zeros(B, 1)
        for b in range(B):
            v = boards[b, 0].item()
            value[b] = 1.0 if v <= 0 else max(0.0, 1.0 - v / 10.0)
        return {"policy": policy, "value": value}


@pytest.fixture
def logic():
    return CountdownLogic()


@pytest.fixture
def model(logic):
    return CountdownModel(logic)


# ---------------------------------------------------------------------------
# PythonPUCT (uses select_leaves_batch + backpropagate_batch)
# ---------------------------------------------------------------------------

class PythonPUCT(PUCT):
    """PUCT using pure-Python kernels."""

    def run_simulation_batch(self, model, active_games, num_simulations=50,
                             c_puct_base=19652, c_puct_init=1.25):
        game_indices = np.array(active_games, dtype=np.int32)
        logic = self.logic

        unexpanded = [self.storage.root_indices[g]
                      for g in active_games
                      if not self.storage.is_expanded[self.storage.root_indices[g]]]
        if unexpanded:
            root_indices = np.array(unexpanded, dtype=np.int32)
            obs_boards = torch.tensor(self.storage.boards[root_indices],
                                      device=self.device, dtype=torch.float32)
            obs_players = torch.tensor(self.storage.players[root_indices],
                                       device=self.device, dtype=torch.long)
            batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
            with torch.no_grad():
                outputs = model.forward_for_mcts(batch)
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[root_indices] = True
            self.storage.prior_probs[root_indices] = priors
            backpropagate_batch(root_indices, vals,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)

        while True:
            min_visits = min(
                self.storage.visit_counts[self.storage.root_indices[g]]
                for g in active_games
            )
            if min_visits >= num_simulations + 1:
                break

            leaf_indices = select_leaves_batch(
                fast_step_func=logic.fast_step,
                get_valid_mask_func=logic.get_valid_mask,
                NUM_ACTIONS=logic.NUM_ACTIONS,
                player1=logic.PLAYER_1, player2=logic.PLAYER_2,
                game_indices=game_indices,
                root_indices=self.storage.root_indices,
                children=self.storage.children,
                visit_counts=self.storage.visit_counts,
                values=self.storage.values,
                prior_probs=self.storage.prior_probs,
                is_expanded=self.storage.is_expanded,
                is_terminal=self.storage.is_terminal,
                terminal_values=self.storage.terminal_values,
                boards=self.storage.boards,
                players=self.storage.players,
                parents=self.storage.parents,
                edge_from_parent=self.storage.edge_from_parent,
                next_free_idx_ptr=self.next_free_idx_arr,
                c_puct_base=c_puct_base, c_puct_init=c_puct_init,
                max_nodes=self.max_nodes,
                depths=self.storage.depths,
                max_game_depth=logic.MAX_MOVES
            )

            is_term = self.storage.is_terminal[leaf_indices]
            leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
            if np.any(is_term):
                leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
            non_term = ~is_term
            if np.any(non_term):
                nn_idx = leaf_indices[non_term]
                obs_b = torch.tensor(self.storage.boards[nn_idx],
                                     device=self.device, dtype=torch.float32)
                obs_p = torch.tensor(self.storage.players[nn_idx],
                                     device=self.device, dtype=torch.long)
                batch = {"boards": obs_b.flatten(1), "current_player": obs_p}
                with torch.no_grad():
                    out = model.forward_for_mcts(batch)
                priors = out['policy'].float().cpu().numpy().astype(np.float64)
                vals = out['value'].float().cpu().numpy().flatten().astype(np.float64)
                self.storage.is_expanded[nn_idx] = True
                self.storage.prior_probs[nn_idx] = priors
                leaf_values[non_term] = vals

            backpropagate_batch(leaf_indices, leaf_values,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)


class PythonGumbelDense(GumbelDense):
    """GumbelDense using pure-Python kernels."""

    def _expand_roots_v4(self, model, active_games):
        n_active = len(active_games)
        root_indices = self.storage.root_indices[active_games]
        obs_boards = torch.tensor(self.storage.boards[root_indices],
                                  device=self.device, dtype=torch.float32)
        obs_players = torch.tensor(self.storage.players[root_indices],
                                   device=self.device, dtype=torch.long)
        batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
        with torch.no_grad():
            outputs = model.forward_for_mcts(batch)
        probs = outputs['policy'].float().cpu().numpy()
        vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
        self.storage.prior_probs[root_indices] = probs
        self.root_logits[:n_active] = np.log(probs + 1e-10)
        self.root_nn_values[:n_active] = vals.astype(np.float32)
        logic = model.logic
        for i in range(n_active):
            r_idx = root_indices[i]
            self.root_legal_masks[i] = logic.get_valid_mask(
                self.storage.boards[r_idx], self.storage.players[r_idx])
        self.storage.is_expanded[root_indices] = True
        backpropagate_batch(root_indices, vals,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def _evaluate_and_backprop_v3(self, model, leaf_indices):
        is_term = self.storage.is_terminal[leaf_indices]
        leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
        if np.any(is_term):
            leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
        non_term = ~is_term
        if np.any(non_term):
            nn_idx = leaf_indices[non_term]
            boards = torch.from_numpy(self.storage.boards[nn_idx]).float()
            players = torch.from_numpy(self.storage.players[nn_idx]).long()
            with torch.no_grad():
                outputs = model.forward_for_mcts(
                    {"boards": boards.flatten(1), "current_player": players})
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[nn_idx] = True
            self.storage.prior_probs[nn_idx] = priors
            leaf_values[non_term] = vals
        backpropagate_batch(leaf_indices, leaf_values,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def run_simulation_batch(self, model, active_games, num_simulations=50):
        game_indices = np.array(active_games, dtype=np.int32)
        logic = model.logic
        n_active = len(active_games)
        self._expand_roots_v4(model, active_games)

        root_idxs = self.storage.root_indices[game_indices]
        raw_probs = self.storage.prior_probs[root_idxs].astype(np.float64)
        legal_f = self.root_legal_masks[:n_active].astype(np.float64)
        legal_probs = raw_probs * legal_f
        legal_probs /= legal_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        mean_ent = float(-(legal_probs * np.log(legal_probs + 1e-10)).sum(axis=1).mean())
        mean_n_legal = float(legal_f.sum(axis=1).mean())
        max_ent = np.log(mean_n_legal) if mean_n_legal > 1.0 else 1.0
        ratio = min(1.0, mean_ent / max(float(max_ent), 1e-8))
        dynamic_max_k = int(round(
            self.max_considered_actions
            - ratio * (self.max_considered_actions - self.entropy_min_k)))
        dynamic_max_k = max(self.entropy_min_k, min(self.max_considered_actions, dynamic_max_k))
        self.last_root_entropy = mean_ent
        self.last_dynamic_max_k = dynamic_max_k

        max_k = min(self.storage.num_actions, dynamic_max_k)
        num_phases = max(1, int(np.log2(max_k)))
        first_phase_budget = num_simulations // num_phases
        k_initial = min(max_k, first_phase_budget // 2)
        k_initial = max(2, k_initial)
        num_phases = max(1, int(np.log2(k_initial)))
        k_initial = min(k_initial, max(2, (num_simulations // num_phases) // 4))
        k_initial = max(2, 1 << int(np.log2(k_initial)))
        num_phases = max(1, int(np.log2(k_initial)))

        candidate_mask = self._get_initial_gumbel_candidates(logic, game_indices, k_initial)
        remaining = num_simulations
        for phase in range(num_phases):
            k_phase = max(1, k_initial // (2 ** phase))
            phases_left = num_phases - phase
            budget_this_phase = remaining if phase == num_phases - 1 else remaining // phases_left
            sims_per_action = max(1, budget_this_phase // k_phase)
            for candidate_rank in range(k_phase):
                root_moves = get_forced_root_moves_kernel(
                    n_active, candidate_mask, candidate_rank)
                for _ in range(sims_per_action):
                    leaf_indices = descend_tree_kernel(
                        logic.fast_step, logic.get_valid_mask,
                        self.storage.num_actions,
                        logic.PLAYER_1, logic.PLAYER_2,
                        game_indices,
                        self.storage.root_indices[game_indices],
                        root_moves,
                        self.storage.children, self.storage.visit_counts,
                        self.storage.values, self.storage.prior_probs,
                        self.storage.is_expanded, self.storage.is_terminal,
                        self.storage.terminal_values, self.storage.boards,
                        self.storage.players, self.storage.parents,
                        self.storage.edge_from_parent, self.next_free_idx_arr,
                        self.max_nodes, self.storage.depths, logic.MAX_MOVES,
                        self.c_visit, self.c_scale)
                    self._evaluate_and_backprop_v3(model, leaf_indices)
            remaining -= (k_phase * sims_per_action)
            if phase < num_phases - 1:
                candidate_mask = self._halve_py(
                    game_indices, candidate_mask, get_gumbel_score_kernel)

        final_moves = self._final_py(
            game_indices, candidate_mask, get_gumbel_score_kernel)
        return final_moves

    def _halve_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values, c_visit=self.c_visit, c_scale=self.c_scale)
        new_mask = np.zeros_like(candidate_mask)
        for i in range(n_active):
            active_moves = np.where(candidate_mask[i])[0]
            if len(active_moves) <= 1:
                new_mask[i] = candidate_mask[i]
                continue
            num_to_keep = max(1, len(active_moves) // 2)
            row_scores = scores[i, active_moves]
            top_indices = np.argsort(row_scores)[-num_to_keep:]
            new_mask[i, active_moves[top_indices]] = True
        return new_mask

    def _final_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values, c_visit=self.c_visit, c_scale=self.c_scale)
        return np.array([np.argmax(scores[i]) for i in range(n_active)], dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests: select_leaves_batch + backpropagate_batch (PUCT path)
# ---------------------------------------------------------------------------

class TestPythonPUCT:
    def test_runs_without_error(self, logic, model):
        tree = PythonPUCT(n_games=1, max_nodes=200, logic=logic, device="cpu")
        board = logic.get_initial_board()
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=16)
        visits, _ = tree.get_all_root_data(n_active=1)
        assert visits[0].sum() > 0

    def test_visits_increase_with_budget(self, logic, model):
        tree = PythonPUCT(n_games=1, max_nodes=500, logic=logic, device="cpu")
        board = logic.get_initial_board()
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=32)
        visits, _ = tree.get_all_root_data(n_active=1)
        total = visits[0].sum()
        assert total >= 32

    def test_finds_optimal_action(self, logic, model):
        """Starting from 3, subtract-3 (action 2) solves immediately."""
        tree = PythonPUCT(n_games=1, max_nodes=500, logic=logic, device="cpu")
        board = np.array([3.0], dtype=np.float64)
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=64)
        visits, _ = tree.get_all_root_data(n_active=1)
        best = int(np.argmax(visits[0]))
        assert best == 2  # subtract 3

    def test_multiple_games(self, logic, model):
        tree = PythonPUCT(n_games=2, max_nodes=500, logic=logic, device="cpu")
        boards = np.array([[5.0], [3.0]], dtype=np.float64)
        tree.initialize_roots([0, 1], boards, np.array([1, 1]))
        tree.run_simulation_batch(model, [0, 1], num_simulations=32)
        visits, _ = tree.get_all_root_data(n_active=2)
        assert visits[0].sum() > 0
        assert visits[1].sum() > 0

    def test_terminal_state(self, logic, model):
        """Board at 0 — already done, no valid moves except no-op."""
        tree = PythonPUCT(n_games=1, max_nodes=100, logic=logic, device="cpu")
        board = np.array([0.0], dtype=np.float64)
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=8)
        visits, _ = tree.get_all_root_data(n_active=1)
        assert visits[0][3] > 0  # no-op should get visits


# ---------------------------------------------------------------------------
# Tests: descend_tree_kernel + get_gumbel_score_kernel (Gumbel path)
# ---------------------------------------------------------------------------

class TestPythonGumbelDense:
    def test_runs_without_error(self, logic, model):
        tree = PythonGumbelDense(n_games=1, max_nodes=200, logic=logic, device="cpu")
        board = logic.get_initial_board()
        tree.initialize_roots([0], board[None], np.array([1]))
        actions = tree.run_simulation_batch(model, [0], num_simulations=16)
        assert 0 <= actions[0] < NUM_ACTIONS

    def test_finds_optimal_action(self, logic, model):
        """From 3, Gumbel should find subtract-3 (action 2)."""
        tree = PythonGumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
        board = np.array([3.0], dtype=np.float64)
        tree.initialize_roots([0], board[None], np.array([1]))
        np.random.seed(42)
        actions = tree.run_simulation_batch(model, [0], num_simulations=32)
        assert actions[0] == 2

    def test_solves_countdown(self, logic, model):
        """Full episode: should reach 0 within a few steps."""
        board = logic.get_initial_board()
        np.random.seed(0)
        for step in range(10):
            tree = PythonGumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], np.array([1]))
            actions = tree.run_simulation_batch(model, [0], num_simulations=32)
            action = int(actions[0])
            _, _, done, board = logic.fast_step(board.copy(), action, 1)
            if done:
                break
        assert done, f"Failed to solve in 10 steps, board={board}"

    def test_multiple_games(self, logic, model):
        tree = PythonGumbelDense(n_games=2, max_nodes=500, logic=logic, device="cpu")
        boards = np.array([[5.0], [2.0]], dtype=np.float64)
        tree.initialize_roots([0, 1], boards, np.array([1, 1]))
        actions = tree.run_simulation_batch(model, [0, 1], num_simulations=16)
        assert len(actions) == 2
        assert all(0 <= a < NUM_ACTIONS for a in actions)

    def test_biased_prior(self, logic):
        """With a strong prior on action 2, Gumbel should still find it."""
        biased = BiasedCountdownModel(logic)
        tree = PythonGumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
        board = np.array([3.0], dtype=np.float64)
        tree.initialize_roots([0], board[None], np.array([1]))
        np.random.seed(42)
        actions = tree.run_simulation_batch(biased, [0], num_simulations=32)
        assert actions[0] == 2


# ---------------------------------------------------------------------------
# Tests: individual kernel functions
# ---------------------------------------------------------------------------

class TestBackpropagateBatch:
    def test_single_leaf(self):
        parents = np.array([-1, 0, 1], dtype=np.int32)
        visit_counts = np.zeros(3, dtype=np.float64)
        values = np.zeros(3, dtype=np.float64)
        leaf_indices = np.array([2], dtype=np.int32)
        nn_values = np.array([0.5], dtype=np.float64)
        backpropagate_batch(leaf_indices, nn_values, parents, visit_counts, values)
        assert visit_counts[2] == 1
        assert visit_counts[1] == 1
        assert visit_counts[0] == 1
        assert values[2] == 0.5
        assert values[1] == 0.5
        assert values[0] == 0.5

    def test_multiple_leaves(self):
        parents = np.array([-1, 0, 0], dtype=np.int32)
        visit_counts = np.zeros(3, dtype=np.float64)
        values = np.zeros(3, dtype=np.float64)
        leaf_indices = np.array([1, 2], dtype=np.int32)
        nn_values = np.array([0.3, 0.7], dtype=np.float64)
        backpropagate_batch(leaf_indices, nn_values, parents, visit_counts, values)
        assert visit_counts[0] == 2
        assert values[0] == pytest.approx(1.0)


class TestGetForcedRootMoves:
    def test_selects_correct_rank(self):
        mask = np.array([[False, True, False, True]], dtype=bool)
        moves = get_forced_root_moves_kernel(1, mask, 0)
        assert moves[0] == 1  # first active
        moves = get_forced_root_moves_kernel(1, mask, 1)
        assert moves[0] == 3  # second active

    def test_wraps_rank(self):
        mask = np.array([[True, False, True, False]], dtype=bool)
        moves = get_forced_root_moves_kernel(1, mask, 2)
        assert moves[0] == 0  # wraps: rank 2 % 2 = 0

    def test_empty_mask(self):
        mask = np.array([[False, False, False, False]], dtype=bool)
        moves = get_forced_root_moves_kernel(1, mask, 0)
        assert moves[0] == 0  # fallback


class TestGetGumbelScore:
    def test_output_shape(self):
        n_active = 1
        num_actions = 4
        game_indices = np.array([0], dtype=np.int32)
        root_indices = np.array([0], dtype=np.int32)
        candidate_mask = np.ones((1, num_actions), dtype=bool)
        children = np.full((1, num_actions), -1, dtype=np.int32)
        visit_counts = np.array([5.0], dtype=np.float64)
        values = np.array([2.0], dtype=np.float64)
        root_logits = np.zeros((1, num_actions), dtype=np.float64)
        gumbel_noises = np.zeros((1, num_actions), dtype=np.float64)
        prior_probs = np.ones((1, num_actions), dtype=np.float64) / num_actions
        nn_values = np.array([0.5], dtype=np.float32)

        scores = get_gumbel_score_kernel(
            n_active, game_indices, root_indices, candidate_mask,
            children, visit_counts, values, root_logits, gumbel_noises,
            prior_probs, nn_values)
        assert scores.shape == (1, num_actions)
        assert np.all(np.isfinite(scores))

    def test_masked_actions_have_low_score(self):
        n_active = 1
        num_actions = 4
        game_indices = np.array([0], dtype=np.int32)
        root_indices = np.array([0], dtype=np.int32)
        candidate_mask = np.array([[True, False, True, False]], dtype=bool)
        children = np.full((1, num_actions), -1, dtype=np.int32)
        visit_counts = np.array([1.0], dtype=np.float64)
        values = np.array([0.0], dtype=np.float64)
        root_logits = np.zeros((1, num_actions), dtype=np.float64)
        gumbel_noises = np.zeros((1, num_actions), dtype=np.float64)
        prior_probs = np.ones((1, num_actions), dtype=np.float64) / num_actions
        nn_values = np.array([0.0], dtype=np.float32)

        scores = get_gumbel_score_kernel(
            n_active, game_indices, root_indices, candidate_mask,
            children, visit_counts, values, root_logits, gumbel_noises,
            prior_probs, nn_values)
        assert scores[0, 1] < -1e9
        assert scores[0, 3] < -1e9
        assert scores[0, 0] > -1e9
        assert scores[0, 2] > -1e9


class TestComputeGumbelPolicy:
    def test_output_is_distribution(self):
        n_active = 1
        num_actions = 4
        root_indices = np.array([0], dtype=np.int32)
        children = np.full((1, num_actions), -1, dtype=np.int32)
        visit_counts = np.array([1.0], dtype=np.float64)
        values = np.array([0.0], dtype=np.float64)
        prior_probs = np.ones((1, num_actions), dtype=np.float64) / num_actions
        root_logits = np.log(prior_probs + 1e-10)
        legal_masks = np.ones((1, num_actions), dtype=np.float64)
        nn_values = np.array([0.0], dtype=np.float32)

        policy = compute_gumbel_policy_kernel(
            n_active, root_indices, children, visit_counts, values,
            prior_probs, root_logits, legal_masks, nn_values)
        assert policy.shape == (1, num_actions)
        assert policy[0].sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(policy >= 0)

    def test_illegal_actions_have_zero_prob(self):
        n_active = 1
        num_actions = 4
        root_indices = np.array([0], dtype=np.int32)
        children = np.full((1, num_actions), -1, dtype=np.int32)
        visit_counts = np.array([1.0], dtype=np.float64)
        values = np.array([0.0], dtype=np.float64)
        prior_probs = np.ones((1, num_actions), dtype=np.float64) / num_actions
        root_logits = np.log(prior_probs + 1e-10)
        legal_masks = np.array([[1.0, 0.0, 1.0, 0.0]])
        nn_values = np.array([0.0], dtype=np.float32)

        policy = compute_gumbel_policy_kernel(
            n_active, root_indices, children, visit_counts, values,
            prior_probs, root_logits, legal_masks, nn_values)
        assert policy[0, 1] == 0.0
        assert policy[0, 3] == 0.0
        assert policy[0, 0] > 0
        assert policy[0, 2] > 0
