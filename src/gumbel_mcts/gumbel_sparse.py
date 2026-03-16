"""
GumbelSparse: Sparse Gumbel MCTS for large action spaces.

Key difference from GumbelDense: edges are stored in a flat list with per-node
offset/count, not in dense (max_nodes, NUM_ACTIONS) arrays. This makes
chess (4672 actions) feasible without >GB of memory per tree.

Memory comparison for 200K nodes:
  Dense: children(200K×4672×4B) + priors(200K×4672×8B) ≈ 10.5 GB
  Sparse: edges(7M × ~14B) ≈ 98 MB  (at avg 35 legal moves/node)

The Gumbel sequential halving logic is identical to GumbelDense.
"""

import numpy as np
import torch
from kernels.puct_kernels import backpropagate_batch
from kernels.gumbel_sparse_kernels import (
    descend_batch,
    descend_batch_python,
    get_forced_edge_local,
    compute_gumbel_scores,
    compute_improved_policy,
)

def _make_rust_descend_wrapper(rust_fn):
    """
    Adapt the Rust mcts_descend_batch to match the Python call signature.
    
    The Rust function:
      - Takes read-only args first, then mutable args (for PyO3)
      - Does NOT take fast_step as a parameter (calls it internally)
      - Has the same semantics as descend_batch_python
    """
    def wrapper(
        fast_step_func,        # ignored — Rust calls fast_step_impl internally
        game_indices,          # (n_active,) int32
        root_indices,          # (n_games,) int32
        forced_edge_locals,    # (n_active,) int32
        # Node arrays
        visit_counts, values, is_expanded, is_terminal, terminal_values,
        boards, players, parents, edge_from_parent, depths, node_nn_value,
        # Edge arrays
        node_edge_offset, node_num_edges,
        edge_action, edge_child, edge_prior,
        # Allocator
        next_free_node_arr, max_nodes,
        # Params
        c_visit, c_scale_vector, max_game_depth,
        player1, player2,
        board_rows=0, board_cols=0,  # ignored by Rust
    ):
        return rust_fn(
            # Read-only inputs
            game_indices,
            root_indices,
            forced_edge_locals,
            node_edge_offset,
            node_num_edges,
            edge_action,
            edge_prior,
            # Mutable node arrays
            visit_counts,
            values,
            is_expanded,
            is_terminal,
            terminal_values,
            boards,
            players,
            parents,
            edge_from_parent,
            depths,
            node_nn_value,
            # Mutable edge arrays
            edge_child,
            # Allocator
            next_free_node_arr,
            # Scalar params
            max_nodes,
            c_visit,
            c_scale_vector,
            max_game_depth,
            player1,
            player2,
        )
    return wrapper

class GumbelSparseStorage:
    """
    Flat-array storage with sparse edge lists.
    
    Node arrays are indexed by node_idx (0..max_nodes-1).
    Edge arrays are indexed by a global edge counter. Each expanded node
    owns a contiguous slice [edge_offset .. edge_offset + num_edges).
    """

    def __init__(self, max_nodes, max_edges, n_games, board_shape, num_actions, logic):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.n_games = n_games
        self.num_actions = num_actions
        self.logic = logic
        self.node_state_width = logic.NODE_STORAGE_WIDTH

        # --- Per-node arrays ---
        self.parents       = np.full(max_nodes, -1, dtype=np.int32)
        self.visit_counts  = np.zeros(max_nodes, dtype=np.int32)
        self.values        = np.zeros(max_nodes, dtype=np.float64)
        self.is_expanded   = np.zeros(max_nodes, dtype=np.bool_)
        self.is_terminal   = np.zeros(max_nodes, dtype=np.bool_)
        self.terminal_values = np.zeros(max_nodes, dtype=np.float64)
        self.boards        = np.zeros((max_nodes, self.node_state_width), dtype=np.int8)
        self.players       = np.zeros(max_nodes, dtype=np.int8)
        self.depths        = np.zeros(max_nodes, dtype=np.int32)
        self.edge_from_parent = np.zeros(max_nodes, dtype=np.int16)
        self.root_indices  = np.zeros(n_games, dtype=np.int32)

        # NN value stored at expansion time (needed for v_mix in Gumbel)
        self.node_nn_value = np.zeros(max_nodes, dtype=np.float32)

        # --- Per-node edge metadata ---
        self.node_edge_offset = np.zeros(max_nodes, dtype=np.int32)
        self.node_num_edges   = np.zeros(max_nodes, dtype=np.int16)

        # --- Flat edge arrays ---
        self.edge_action = np.zeros(max_edges, dtype=np.int16)       # global action idx
        self.edge_child  = np.full(max_edges, -1, dtype=np.int32)    # child node idx
        self.edge_prior  = np.zeros(max_edges, dtype=np.float64)     # P(s,a)

class GumbelSparse:
    """
    Gumbel MCTS with sparse edge storage.
    
    Drop-in replacement for GumbelDense — same external API:
      - run_simulation_batch(model, active_games, num_simulations) -> moves
      - get_improved_policy(n_active) -> (n_active, NUM_ACTIONS) dense
      - get_gumbel_root_value(n_active, chosen_moves) -> (n_active,) float
    """

    def __init__(self, n_games, max_nodes, logic, device='cuda',
                 c_visit=50.0, c_scale=1.0,
                 avg_branching=35, max_legal_moves=256):
        """
        Args:
            n_games: number of simultaneous games
            max_nodes: max tree nodes across all games
            logic: game logic module (provides BOARD_SHAPE, NUM_ACTIONS, fast_step, etc.)
            device: torch device for NN inference
            c_visit, c_scale: Gumbel MCTS exploration constants
            avg_branching: expected average legal moves per position (for edge allocation)
            max_legal_moves: max possible legal moves in any position (e.g. 218 for chess)
        """
        max_edges = max_nodes * avg_branching
        self.logic = logic

        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.n_games = n_games
        self.device = device
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.max_legal = max_legal_moves

        self.next_free_node_arr = np.array([1], dtype=np.int32)
        self.next_free_edge_arr = np.array([0], dtype=np.int32)

        # Board reshape params for descent kernels
        board_shape = logic.BOARD_SHAPE
        if len(board_shape) == 2:
            self._board_rows, self._board_cols = board_shape
        else:
            # 1D board (e.g. chess (69,)) — reshape(1, N) is a no-op for flat arrays
            self._board_rows, self._board_cols = 1, board_shape[0]

        # --- Root-level Gumbel data (padded to max_legal) ---
        self.root_logits      = np.zeros((n_games, max_legal_moves), dtype=np.float32)
        self.root_gumbel_noise = np.zeros((n_games, max_legal_moves), dtype=np.float32)
        self.root_actions     = np.full((n_games, max_legal_moves), -1, dtype=np.int16)
        self.root_num_legal   = np.zeros(n_games, dtype=np.int16)
        self.candidate_mask   = np.zeros((n_games, max_legal_moves), dtype=np.bool_)

        # --- Storage (eager init) ---
        self.storage = GumbelSparseStorage(
            max_nodes=max_nodes,
            max_edges=max_edges,
            n_games=n_games,
            board_shape=logic.BOARD_SHAPE,
            num_actions=logic.NUM_ACTIONS,
            logic=logic
        )

        # Autocast settings
        if device == 'cuda' or (isinstance(device, str) and 'cuda' in device):
            self._autocast_device = 'cuda'
            self._use_autocast = True
            self._autocast_dtype = torch.bfloat16
        elif device == 'mps':
            self._autocast_device = 'mps'
            self._use_autocast = False
            self._autocast_dtype = torch.float16
        else:
            self._autocast_device = 'cpu'
            self._use_autocast = False
            self._autocast_dtype = torch.float16

    def _batch_get_legal_masks(self, node_indices):
        """
        Compute legal masks for multiple nodes in a single batch call.
        
        Uses logic.get_legal_masks_batch (numpy-native) if available (e.g. chess_core),
        otherwise falls back to per-node logic.get_valid_mask loop.
        
        Args:
            node_indices: (N,) int32 array of node indices
            
        Returns:
            masks: (N, NUM_ACTIONS) bool numpy array
        """
        s = self.storage
        logic = self.logic
        n = len(node_indices)

        if n == 0:
            return np.zeros((0, logic.NUM_ACTIONS), dtype=np.bool_)

        # Fast path: batch call (chess_core, or any logic with numpy batch support)
        if hasattr(logic, 'get_legal_masks_batch'):
            if logic.USE_HISTORY:
                boards = s.boards[node_indices, logic.NN_OBS_WIDTH:]
            else:
                boards = s.boards[node_indices]
            # Flatten if board shape is multi-dimensional (e.g. (8,8) for othello)
            if len(logic.BOARD_SHAPE) > 1:
                boards = boards.reshape(n, -1)
            players = s.players[node_indices].astype(np.int64)
            return logic.get_legal_masks_batch(boards, players)

        # Slow path: per-node loop
        masks = np.zeros((n, logic.NUM_ACTIONS), dtype=np.bool_)
        start_idx = logic.NN_OBS_WIDTH if logic.USE_HISTORY else 0

        for j, node_idx in enumerate(node_indices):
            raw_board = s.boards[node_idx, start_idx:].reshape(logic.BOARD_SHAPE)
            masks[j] = logic.get_valid_mask(raw_board, s.players[node_idx])
        return masks

    def reset(self):
        num_used = self.next_free_node_arr[0]
        self.next_free_node_arr[0] = 1
        self.next_free_edge_arr[0] = 0
        s = self.storage
        s.depths[:num_used + 1] = 0
        s.is_expanded[:num_used + 1] = False
        s.is_terminal[:num_used + 1] = False

    def initialize_roots(self, active_games, starting_boards, starting_players):
        current_alloc = self.next_free_node_arr[0]
        s = self.storage
        for i, game_idx in enumerate(active_games):
            root_idx = current_alloc + i
            s.root_indices[game_idx] = root_idx
            s.boards[root_idx] = starting_boards[i]
            s.players[root_idx] = starting_players[i]
            s.parents[root_idx] = -1
            s.visit_counts[root_idx] = 0
            s.values[root_idx] = 0.0
            s.is_expanded[root_idx] = False
            s.is_terminal[root_idx] = False
            s.depths[root_idx] = 0
        self.next_free_node_arr[0] += len(active_games)

    def _get_descend_fn(self):
        """Pick the fastest available descent implementation.
        
        Priority: Rust (chess_core) > Numba (@njit) > pure Python.
        
        The Rust path eliminates all Python↔Rust boundary crossings inside
        the descent loop — fast_step is called directly within Rust.
        """
        if not hasattr(self, '_descend_fn'):
            # 1. Try Rust descent (chess-specific, fastest)
            rust_fn = getattr(self.logic, 'mcts_descend_batch', None)
            if rust_fn is not None:
                self._descend_fn = _make_rust_descend_wrapper(rust_fn)
            else:
                # 2. Try Numba (for games with @njit fast_step)
                try:
                    from numba.core.dispatcher import Dispatcher
                    if isinstance(self.logic.fast_step, Dispatcher):
                        self._descend_fn = descend_batch
                    else:
                        self._descend_fn = descend_batch_python
                except ImportError:
                    # 3. Fallback to pure Python
                    self._descend_fn = descend_batch_python
        return self._descend_fn

    def run_simulation_batch(self, model, active_games, num_simulations=50, c_scale_overrides=None):
        """
        Run Gumbel MCTS with sequential halving.
        Returns chosen move (global action index) per active game.
        c_scale_overrides lets us use different c_scale for opening vs late game (e.g. chess) if desired.
        """
        logic = self.logic

        game_indices = np.array(active_games, dtype=np.int32)
        n_active = len(active_games)

        # Convert overrides to a usable numpy format if provided
        if c_scale_overrides is not None:
            if isinstance(c_scale_overrides, torch.Tensor):
                current_c_scales = c_scale_overrides.detach().cpu().numpy()
            else:
                current_c_scales = np.asarray(c_scale_overrides)
        else:
            current_c_scales = np.full(n_active, self.c_scale, dtype=np.float32)

        # 1. EXPAND ROOTS (get logits + priors)
        self._expand_roots(model, active_games)

        # 2. SEQUENTIAL HALVING SETUP
        max_k = min(self.max_legal, 16)
        num_phases = max(1, int(np.log2(max_k)))
        first_budget = num_simulations // num_phases
        k_initial = min(max_k, first_budget // 2)
        k_initial = max(2, k_initial)
        num_phases = max(1, int(np.log2(k_initial)))

        # Select initial candidates: top-k by (logit + gumbel_noise)
        self._init_candidates(n_active, k_initial)

        # 3. HALVING LOOP
        remaining = num_simulations
        for phase in range(num_phases):
            k_phase = max(1, k_initial // (2 ** phase))
            phases_left = num_phases - phase
            budget = remaining if phase == num_phases - 1 else remaining // phases_left
            sims_per_action = max(1, budget // k_phase)

            for candidate_rank in range(k_phase):
                forced_edges = get_forced_edge_local(
                    self.candidate_mask[:n_active], candidate_rank
                )

                for _ in range(sims_per_action):
                    leaf_indices = self._get_descend_fn()(
                        logic.fast_step,
                        game_indices,
                        self.storage.root_indices,
                        forced_edges,
                        # Node arrays
                        self.storage.visit_counts, self.storage.values,
                        self.storage.is_expanded, self.storage.is_terminal,
                        self.storage.terminal_values,
                        self.storage.boards, self.storage.players,
                        self.storage.parents, self.storage.edge_from_parent,
                        self.storage.depths, self.storage.node_nn_value,
                        # Edge arrays
                        self.storage.node_edge_offset, self.storage.node_num_edges,
                        self.storage.edge_action, self.storage.edge_child,
                        self.storage.edge_prior,
                        # Allocator
                        self.next_free_node_arr, self.max_nodes,
                        # Params
                        self.c_visit, current_c_scales, logic.MAX_MOVES,
                        logic.PLAYER_1, logic.PLAYER_2,
                        self._board_rows, self._board_cols,
                    )

                    self._evaluate_and_expand(model, leaf_indices)

            remaining -= k_phase * sims_per_action

            # Halve candidates
            if phase < num_phases - 1:
                self._halve_candidates(n_active, current_c_scales)

        # 4. PICK FINAL MOVES
        return self._get_final_moves(n_active, current_c_scales)

    # =========================================================================
    # Root expansion
    # =========================================================================

    def _expand_roots(self, model, active_games):
        """Expand root nodes: NN eval → allocate edges → store logits + Gumbel noise."""
        n_active = len(active_games)
        s = self.storage
        logic = self.logic
        root_indices = s.root_indices[active_games]

        # Compute legal masks first (from raw board shadow, not NN observation)
        all_masks = self._batch_get_legal_masks(root_indices)

        # NN inference
        nn_width = self.logic.NN_OBS_WIDTH
        boards_t = torch.tensor(s.boards[root_indices, :nn_width], device=self.device, dtype=torch.float32)
        players_t = torch.tensor(s.players[root_indices], device=self.device, dtype=torch.long)
        masks_t = torch.from_numpy(all_masks).to(self.device)
        batch = {"boards": boards_t, "current_player": players_t, "legal_actions_mask": masks_t}

        with torch.no_grad():
            if self._use_autocast:
                with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                    outputs = model.forward_for_mcts(batch)
            else:
                outputs = model.forward_for_mcts(batch)

        probs = outputs['policy'].float().cpu().numpy()      # (n_active, NUM_ACTIONS) dense
        vals = outputs['value'].float().cpu().numpy().flatten()

        # Allocate edges + store logits for each root
        for i in range(n_active):
            r_idx = root_indices[i]

            legal_moves = np.where(all_masks[i])[0].astype(np.int16)
            n_legal = len(legal_moves)

            # Allocate edges
            self._allocate_edges(r_idx, legal_moves, probs[i])

            # Store NN value for v_mix
            s.node_nn_value[r_idx] = vals[i]

            # Root-level Gumbel data (padded)
            self.root_num_legal[i] = n_legal
            self.root_actions[i, :n_legal] = legal_moves
            self.root_actions[i, n_legal:] = -1

            legal_probs = probs[i, legal_moves]
            self.root_logits[i, :n_legal] = np.log(np.maximum(legal_probs, 1e-10))
            self.root_logits[i, n_legal:] = -1e18

            # Sample Gumbel noise: G = -log(-log(U))
            # Clamp away from {0, 1} to avoid inf/nan from float32 rounding at boundaries.
            u = np.random.uniform(0.0, 1.0, size=n_legal).astype(np.float32)
            u = np.clip(u, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
            self.root_gumbel_noise[i, :n_legal] = -np.log(-np.log(u))
            self.root_gumbel_noise[i, n_legal:] = -1e18

        # Mark expanded & backprop initial values
        s.is_expanded[root_indices] = True
        backpropagate_batch(
            root_indices, vals.astype(np.float64),
            s.parents, s.visit_counts, s.values
        )

    def _allocate_edges(self, node_idx, legal_moves, dense_probs):
        """
        Allocate a contiguous block of edges for a node.
        legal_moves: (n_legal,) int16 — global action indices
        dense_probs: (NUM_ACTIONS,) — raw NN probabilities
        """
        s = self.storage
        n_legal = len(legal_moves)
        start = self.next_free_edge_arr[0]

        if start + n_legal > self.max_edges:
            raise RuntimeError(
                f"Edge pool exhausted: need {start + n_legal}, have {self.max_edges}. "
                f"Increase avg_branching or max_nodes."
            )

        s.node_edge_offset[node_idx] = start
        s.node_num_edges[node_idx] = n_legal

        s.edge_action[start:start + n_legal] = legal_moves
        s.edge_child[start:start + n_legal] = -1

        # Extract and renormalize priors for legal moves only
        legal_priors = dense_probs[legal_moves].astype(np.float64)
        prior_sum = legal_priors.sum()
        if prior_sum > 1e-10:
            legal_priors /= prior_sum
        else:
            legal_priors[:] = 1.0 / max(n_legal, 1)
        s.edge_prior[start:start + n_legal] = legal_priors

        self.next_free_edge_arr[0] = start + n_legal

    # =========================================================================
    # Leaf evaluation & expansion
    # =========================================================================
    
    def _evaluate_and_expand(self, model, leaf_indices):
        """Evaluate leaves with NN, allocate edges for non-terminals, backprop."""
        s = self.storage

        is_term = s.is_terminal[leaf_indices]
        leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)

        # Terminals: use ground truth
        if np.any(is_term):
            leaf_values[is_term] = s.terminal_values[leaf_indices[is_term]]

        # Non-terminals: NN eval + expand
        non_term = ~is_term & ~s.is_expanded[leaf_indices]
        if np.any(non_term):
            nn_indices = leaf_indices[non_term]
            nn_width = self.logic.NN_OBS_WIDTH

            # Compute legal masks first (from raw board shadow, not NN observation)
            all_masks = self._batch_get_legal_masks(nn_indices)

            # Slice the node memory. 
            # For Chess, this takes indices [0:515], leaving the raw shadow [515:584] behind.
            raw_boards = s.boards[nn_indices, :nn_width]

            # NN inference (already batched — no change)
            boards_t = torch.from_numpy(raw_boards).to(self.device).float()
            players_t = torch.from_numpy(s.players[nn_indices]).to(self.device).long()
            masks_t = torch.from_numpy(all_masks).to(self.device)

            with torch.no_grad():
                # Inference expects flattened boards: (B, STATE_VECTOR_LEN)
                batch = {"boards": boards_t, "current_player": players_t, "legal_actions_mask": masks_t}
                if self._use_autocast:
                    with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                        outputs = model.forward_for_mcts(batch)
                else:
                    outputs = model.forward_for_mcts(batch)

            probs = outputs['policy'].float().cpu().numpy()
            vals = outputs['value'].float().cpu().numpy().flatten()

            # Expand each leaf: allocate edges (still loops for bookkeeping)
            for j, node_idx in enumerate(nn_indices):
                legal_moves = np.where(all_masks[j])[0].astype(np.int16)
                self._allocate_edges(node_idx, legal_moves, probs[j])
                s.node_nn_value[node_idx] = vals[j]

            s.is_expanded[nn_indices] = True
            leaf_values[non_term] = vals.astype(np.float64)

        backpropagate_batch(
            leaf_indices, leaf_values,
            s.parents, s.visit_counts, s.values
        )

    # =========================================================================
    # Gumbel sequential halving
    # =========================================================================

    def _init_candidates(self, n_active, k):
        """Select top-k candidates by (logit + gumbel_noise) for each game."""
        self.candidate_mask[:n_active] = False
        for i in range(n_active):
            n_legal = self.root_num_legal[i]
            if n_legal == 0:
                continue
            scores = self.root_logits[i, :n_legal] + self.root_gumbel_noise[i, :n_legal]
            k_actual = min(k, n_legal)
            top_k = np.argsort(scores)[-k_actual:]
            self.candidate_mask[i, top_k] = True

    def _halve_candidates(self, n_active, scale_vector):
        """Halve candidates using sigma * q_completed + logit + gumbel scoring."""
        s = self.storage
        scores = compute_gumbel_scores(
            n_active, s.root_indices[:n_active],
            s.node_edge_offset, s.node_num_edges,
            s.edge_child, s.edge_action, s.edge_prior,
            s.visit_counts, s.values, s.node_nn_value,
            self.root_logits[:n_active],
            self.root_gumbel_noise[:n_active],
            self.root_num_legal[:n_active],
            self.candidate_mask[:n_active],
            self.c_visit, scale_vector,
        )

        new_mask = np.zeros_like(self.candidate_mask[:n_active])
        for i in range(n_active):
            active_edges = np.where(self.candidate_mask[i])[0]
            if len(active_edges) <= 1:
                new_mask[i] = self.candidate_mask[i]
                continue
            num_keep = max(1, len(active_edges) // 2)
            edge_scores = scores[i, active_edges]
            top = np.argsort(edge_scores)[-num_keep:]
            new_mask[i, active_edges[top]] = True

        self.candidate_mask[:n_active] = new_mask

    def _get_final_moves(self, n_active, scale_vector):
        """Pick the best surviving candidate per game. Returns global action indices."""
        s = self.storage
        scores = compute_gumbel_scores(
            n_active, s.root_indices[:n_active],
            s.node_edge_offset, s.node_num_edges,
            s.edge_child, s.edge_action, s.edge_prior,
            s.visit_counts, s.values, s.node_nn_value,
            self.root_logits[:n_active],
            self.root_gumbel_noise[:n_active],
            self.root_num_legal[:n_active],
            self.candidate_mask[:n_active],
            self.c_visit, scale_vector,
        )

        moves = np.zeros(n_active, dtype=np.int32)
        for i in range(n_active):
            best_local = np.argmax(scores[i])
            # Map local edge index -> global action
            r_idx = s.root_indices[i]
            eidx = s.node_edge_offset[r_idx] + best_local
            moves[i] = s.edge_action[eidx]
        return moves

    # =========================================================================
    # Public API (matches v4 interface for game_gen.py compatibility)
    # =========================================================================

    def get_improved_policy(self, n_active):
        """
        Returns (n_active, NUM_ACTIONS) dense improved policy for training.
        pi'(a) = softmax(logit(a) + sigma * q_completed(a))
        """
        c_scale_vec = np.full(n_active, self.c_scale, dtype=np.float32)
        s = self.storage
        return compute_improved_policy(
            n_active, s.root_indices[:n_active],
            s.node_edge_offset, s.node_num_edges,
            s.edge_child, s.edge_action, s.edge_prior,
            s.visit_counts, s.values, s.node_nn_value,
            self.root_logits[:n_active],
            self.root_num_legal[:n_active],
            self.c_visit, c_scale_vec,
            s.num_actions,
        )

    def get_gumbel_root_value(self, n_active, chosen_moves=None):
        """Root Q value for value targets. chosen_moves are global action indices."""
        s = self.storage
        root_indices = s.root_indices[:n_active]
        root_values = np.zeros(n_active, dtype=np.float32)

        for i in range(n_active):
            r_idx = root_indices[i]
            e_start = s.node_edge_offset[r_idx]
            n_edges = s.node_num_edges[r_idx]

            if chosen_moves is not None:
                target_action = chosen_moves[i]
                # Find the edge for this action
                for e in range(n_edges):
                    if s.edge_action[e_start + e] == target_action:
                        c_idx = s.edge_child[e_start + e]
                        if c_idx != -1 and s.visit_counts[c_idx] > 0:
                            root_values[i] = -s.values[c_idx] / s.visit_counts[c_idx]
                        break
            else:
                # Fallback: best-visited child
                best_n = 0
                for e in range(n_edges):
                    c_idx = s.edge_child[e_start + e]
                    if c_idx != -1:
                        n_c = s.visit_counts[c_idx]
                        if n_c > best_n:
                            best_n = n_c
                            root_values[i] = -s.values[c_idx] / n_c

        return root_values

    def get_all_root_data(self, n_active):
        """Backward compat: return (n_active, NUM_ACTIONS) visit counts + root Q."""
        s = self.storage
        root_indices = s.root_indices[:n_active]
        visits = np.zeros((n_active, s.num_actions), dtype=np.float32)
        root_q = np.zeros(n_active, dtype=np.float32)

        for i in range(n_active):
            r_idx = root_indices[i]
            e_start = s.node_edge_offset[r_idx]
            n_edges = s.node_num_edges[r_idx]
            best_n, best_q = 0, 0.0

            for e in range(n_edges):
                c_idx = s.edge_child[e_start + e]
                if c_idx != -1:
                    n_c = s.visit_counts[c_idx]
                    action = s.edge_action[e_start + e]
                    visits[i, action] = n_c
                    if n_c > best_n:
                        best_n = n_c
                        best_q = -s.values[c_idx] / n_c
            root_q[i] = best_q

        return visits, root_q

    def get_max_depth(self):
        upper = self.next_free_node_arr[0]
        if upper <= 1:
            return 0
        return int(np.max(self.storage.depths[:upper]))