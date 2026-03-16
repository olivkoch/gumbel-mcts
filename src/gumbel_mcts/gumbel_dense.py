import numpy as np
import torch
from gumbel_mcts.puct import PUCT
from kernels.puct_kernels import backpropagate_batch
from kernels.gumbel_dense_kernels import descend_tree_kernel, get_forced_root_moves_kernel, get_gumbel_score_kernel, compute_gumbel_policy_kernel

class GumbelDense(PUCT):
    """Gumbel MCTS implementation."""
    
    def __init__(self, n_games, max_nodes, logic, device='cuda', c_visit=50.0, c_scale=1.0):
        super().__init__(n_games, max_nodes, logic, device)
        # We need to store the Gumbel noise for the duration of the search
        self.gumbel_noises = np.zeros((n_games, self.storage.num_actions), dtype=np.float32)
        # Store raw logits for the Muzero formula
        self.root_logits = np.zeros((n_games, self.storage.num_actions), dtype=np.float32)
        self.root_legal_masks = np.zeros((n_games, self.storage.num_actions), dtype=np.bool_)
    
        self.root_nn_values = np.zeros(n_games, dtype=np.float32)
        
        self.c_visit = c_visit
        self.c_scale = c_scale

    def initialize_roots(self, active_games, starting_boards, starting_players):
        super().initialize_roots(active_games, starting_boards, starting_players)
        # Sample fresh Gumbel noise for each game's root
        # G = -log(-log(U)) where U ~ Uniform(0,1)
        u = np.random.uniform(0, 1, size=(len(active_games), self.storage.num_actions)).astype(np.float32)
        u = np.clip(u, np.finfo(np.float32).tiny, 1.0 - np.finfo(np.float32).eps)
        self.gumbel_noises[:len(active_games)] = -np.log(-np.log(u))

    def run_simulation_batch(self, model, active_games, num_simulations=50):
        game_indices = np.array(active_games, dtype=np.int32)
        logic = model.logic
        n_active = len(active_games)

        # 1. INITIAL ROOT EXPANSION (To get Logits)
        # We need raw logits for Gumbel. Most models output softmax.
        # Ensure your model has a way to return raw scores or apply np.log(probs)
        self._expand_roots_v4(model, active_games)

        # 2. SETUP SEQUENTIAL HALVING
        # k_initial = min(self.storage.num_actions, 16)
        # num_phases = max(1, int(np.log2(k_initial)))
        max_k = min(self.storage.num_actions, 16)
        num_phases = max(1, int(np.log2(max_k)))

        # Ensure first phase gives ≥ 2 sims per candidate
        first_phase_budget = num_simulations // num_phases
        k_initial = min(max_k, first_phase_budget // 2)
        k_initial = max(2, k_initial)  # need at least 2 to halve

        # Recompute phases for actual k
        num_phases = max(1, int(np.log2(k_initial)))

        # Initialize Candidate Mask (all legal moves are candidates)
        candidate_mask = self._get_initial_gumbel_candidates(logic, game_indices, k_initial)

        # 3. THE HALVING LOOP
        remaining = num_simulations
        for phase in range(num_phases):
            k_phase = max(1, k_initial // (2 ** phase))
            phases_left = num_phases - phase
            if phase == num_phases - 1:
                budget_this_phase = remaining  # Use all remaining sims in the last phase
            else:
                budget_this_phase = remaining // phases_left
            sims_per_action = max(1, budget_this_phase // k_phase)

            # A. Execute the budget for this phase
            for candidate_rank in range(k_phase):

                root_moves = get_forced_root_moves_kernel(
                    n_active, candidate_mask, candidate_rank
                )
                
                for _ in range(sims_per_action):

                    leaf_indices = descend_tree_kernel(
                        logic.fast_step, logic.get_valid_mask, self.storage.num_actions,
                        logic.PLAYER_1, logic.PLAYER_2,
                        game_indices, self.storage.root_indices[game_indices], root_moves,
                        self.storage.children, self.storage.visit_counts, self.storage.values,
                        self.storage.prior_probs, self.storage.is_expanded, self.storage.is_terminal,
                        self.storage.terminal_values, self.storage.boards, self.storage.players,
                        self.storage.parents, self.storage.edge_from_parent, self.next_free_idx_arr,
                        self.max_nodes, self.storage.depths, logic.MAX_MOVES, 
                        self.c_visit, self.c_scale
                    )

                    self._evaluate_and_backprop_v3(model, leaf_indices)

            remaining -= (k_phase * sims_per_action)

            # C. HALVE CANDIDATES
            if phase < num_phases - 1:
                candidate_mask = self._halve_candidates(game_indices, candidate_mask)

        final_moves = self._get_final_survivors(game_indices, candidate_mask)
        return final_moves

    def _halve_candidates(self, game_indices, candidate_mask):
        n_active = len(game_indices)
        
        # 1. Get unified scores from Numba
        scores = get_gumbel_score_kernel(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs, self.root_nn_values, 
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        
        # 2. Prune
        new_mask = np.zeros_like(candidate_mask)
        for i in range(n_active):
            active_moves = np.where(candidate_mask[i])[0]
            if len(active_moves) <= 1:
                new_mask[i] = candidate_mask[i]
                continue
            
            num_to_keep = max(1, len(active_moves) // 2)
            # Use the scores we just calculated
            row_scores = scores[i, active_moves]
            top_indices = np.argsort(row_scores)[-num_to_keep:]
            new_mask[i, active_moves[top_indices]] = True
        return new_mask

    def _get_final_survivors(self, game_indices, candidate_mask):
        n_active = len(game_indices)
        
        # Reuse the same scoring kernel for consistency
        scores = get_gumbel_score_kernel(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs, self.root_nn_values, 
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        
        survivors = np.zeros(n_active, dtype=np.int32)
        for i in range(n_active):
            # Best score among the survivors of the last halving phase
            survivors[i] = np.argmax(scores[i])
        return survivors
    
    def _compute_gumbel_policy(self, root_logits, root_indices, n_active):
        """
        For backward compatibility
        """
        return self.get_improved_policy(n_active)
    
    def _expand_roots_v4(self, model, active_games):
        """Initial expansion of root nodes to populate root_logits."""
        n_active = len(active_games)
        root_indices = self.storage.root_indices[active_games]
        
        # 1. Prepare batch
        leaves_boards = self.storage.boards[root_indices]
        leaves_players = self.storage.players[root_indices]
        
        obs_boards = torch.tensor(leaves_boards, device=self.device, dtype=torch.float32)
        obs_players = torch.tensor(leaves_players, device=self.device, dtype=torch.long)

        batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
        
        # 2. Inference
        with torch.no_grad():
            if self._use_autocast:
                with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                    outputs = model.forward_for_mcts(batch)
            else:
                outputs = model.forward_for_mcts(batch)
        
        # 3. Store Logits
        # If model outputs policy (probs), convert to log-space
        probs = outputs['policy'].float().cpu().numpy()
        vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
        
        self.storage.prior_probs[root_indices] = probs

        self.root_logits[:n_active] = np.log(probs + 1e-10) 
        
        # Cache NN value estimates for v_mix
        self.root_nn_values[:n_active] = vals.astype(np.float32)

        logic = model.logic

        for i in range(n_active):
            r_idx = root_indices[i]
            self.root_legal_masks[i] = logic.get_valid_mask(
                self.storage.boards[r_idx], self.storage.players[r_idx]
            )
            
        # 4. Mark as expanded and store initial value
        self.storage.is_expanded[root_indices] = True
        
        # Backpropagate initial root values
        backpropagate_batch(
            root_indices, vals,
            self.storage.parents, self.storage.visit_counts, self.storage.values
        )
    
    def _evaluate_and_backprop_v3(self, model, leaf_indices):
        """Standard evaluation and backpropagation for a batch of leaves."""
        is_term = self.storage.is_terminal[leaf_indices]
        leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
        
        # A. Handle Terminals
        if np.any(is_term):
            leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
        
        # B. Handle Non-Terminals via NN
        non_term_mask = ~is_term
        if np.any(non_term_mask):
            nn_indices = leaf_indices[non_term_mask]
            
            # Prepare tensors
            boards = torch.from_numpy(self.storage.boards[nn_indices]).to(self.device).float()
            players = torch.from_numpy(self.storage.players[nn_indices]).to(self.device).long()
            
            with torch.no_grad():
                if self._use_autocast:
                    with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                        outputs = model.forward_for_mcts({"boards": boards.flatten(1), "current_player": players})
                else:
                    outputs = model.forward_for_mcts({"boards": boards.flatten(1), "current_player": players})
                
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            
            self.storage.is_expanded[nn_indices] = True
            self.storage.prior_probs[nn_indices] = priors
            leaf_values[non_term_mask] = vals
            
        backpropagate_batch(
            leaf_indices, leaf_values,
            self.storage.parents, self.storage.visit_counts, self.storage.values
        )

    def _get_initial_gumbel_candidates(self, logic, game_indices, k):
        n_active = len(game_indices)
        mask = np.zeros((n_active, self.storage.num_actions), dtype=np.bool_)
        for i in range(n_active):

            scores = self.root_logits[i] + self.gumbel_noises[i]
            legal_mask = self.root_legal_masks[i]
            
            # Only score legal moves
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                continue
            
            # Pick top-k from legal moves only
            k_actual = min(k, len(legal_indices))
            legal_scores = scores[legal_indices]
            top_k_local = np.argsort(legal_scores)[-k_actual:]
            top_k_global = legal_indices[top_k_local]
            
            mask[i, top_k_global] = True
        return mask
    
    def get_gumbel_root_value(self, n_active, chosen_moves=None):
        root_indices = self.storage.root_indices[:n_active]
        
        if chosen_moves is not None:
            # Gather the child index for each game's chosen move
            child_indices = self.storage.children[root_indices, chosen_moves]
            valid = child_indices != -1
            visits = np.where(valid, self.storage.visit_counts[np.maximum(child_indices, 0)], 0)
            values = np.where(valid, self.storage.values[np.maximum(child_indices, 0)], 0.0)
            
            root_values = np.where(
                visits > 0,
                -values / visits,
                0.0
            )
            return root_values.astype(np.float32)
        
        # Fallback: best-visited child per root
        all_children = self.storage.children[root_indices]  # (n_active, num_actions)
        valid = all_children != -1
        safe_idx = np.maximum(all_children, 0)
        visits = np.where(valid, self.storage.visit_counts[safe_idx], 0)
        values = np.where(valid, self.storage.values[safe_idx], 0.0)
        
        best_action = np.argmax(visits, axis=1)
        best_visits = visits[np.arange(n_active), best_action]
        best_values = values[np.arange(n_active), best_action]
        
        return np.where(best_visits > 0, -best_values / best_visits, 0.0).astype(np.float32)
    
    def get_improved_policy(self, n_active):
        """
        Returns the improved policy π' = softmax(logits + σ(completedQ))
        for training. Call this after search completes.
        """
        root_indices = self.storage.root_indices[:n_active]
        
        return compute_gumbel_policy_kernel(
            n_active,
            root_indices,
            self.storage.children,
            self.storage.visit_counts,
            self.storage.values,
            self.storage.prior_probs,
            self.root_logits[:n_active],
            self.root_legal_masks[:n_active],
            self.root_nn_values[:n_active],
            self.c_visit,
            self.c_scale
        )
    
    def get_all_root_data(self, n_active):
        """For backward compatibility with visit-count based policies."""
        root_indices = self.storage.root_indices[:n_active]
        
        visits = np.zeros((n_active, self.storage.num_actions), dtype=np.float32)
        root_q = np.zeros(n_active, dtype=np.float32)
        
        for i in range(n_active):
            r_idx = root_indices[i]
            best_n = 0
            best_q = 0.0
            
            for a in range(self.storage.num_actions):
                c_idx = self.storage.children[r_idx, a]
                if c_idx != -1:
                    n_c = self.storage.visit_counts[c_idx]
                    visits[i, a] = n_c
                    if n_c > best_n:
                        best_n = n_c
                        best_q = -self.storage.values[c_idx] / n_c
            
            root_q[i] = best_q
        
        return visits, root_q