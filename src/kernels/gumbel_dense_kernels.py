import numpy as np
from numba import njit
from kernels.puct_kernels import _init_node

@njit(cache=True)
def get_gumbel_score_kernel(
    n_active, game_indices, root_indices, candidate_mask, 
    children, visit_counts, values, root_logits, gumbel_noises,
    prior_probs, nn_values,
    c_visit=50, c_scale=1.0
):
    num_actions = root_logits.shape[1]
    scores = np.full((n_active, num_actions), -1e10, dtype=np.float32)
    
    for i in range(n_active):
        g_idx = game_indices[i]
        r_idx = root_indices[g_idx]
        
        max_n = 0
        q_min, q_max = 1e10, -1e10
        
        # Compute v_mix (Eq. 33)
        v_hat = nn_values[i]
        sum_n = visit_counts[r_idx] - 1  # exclude root's own visit
        sum_weighted_q = 0.0
        sum_pi_visited = 0.0
        
        for m in range(num_actions):
            c_idx = children[r_idx, m]
            if c_idx != -1:
                n_c = visit_counts[c_idx]
                if n_c > max_n:
                    max_n = n_c
                if n_c > 0:
                    q_c = -values[c_idx] / n_c
                    q_min = min(q_min, q_c)
                    q_max = max(q_max, q_c)
                    
                    pi_a = prior_probs[r_idx, m]
                    sum_weighted_q += pi_a * q_c
                    sum_pi_visited += pi_a
        
        if sum_pi_visited > 1e-10 and sum_n > 0:
            v_mix = (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
        else:
            v_mix = v_hat
        
        # Include v_mix in normalization range
        q_min = min(q_min, v_mix)
        q_max = max(q_max, v_mix)
        q_range = q_max - q_min
        if q_range < 1e-6:
            q_range = 1.0
        
        dynamic_q_scale = (c_visit + max_n) * c_scale
        
        for move in range(num_actions):
            if not candidate_mask[i, move]:
                continue
            
            child_idx = children[r_idx, move]
            n_v = visit_counts[child_idx] if child_idx != -1 else 0
            q_v = (-values[child_idx] / n_v) if n_v > 0 else v_mix
            
            q_normalized = (q_v - q_min) / q_range
            
            scores[i, move] = (root_logits[g_idx, move] + 
                  gumbel_noises[g_idx, move] + 
                  dynamic_q_scale * q_normalized)
            
    return scores

@njit(cache=False)
def descend_tree_kernel(
    fast_step_func, get_valid_mask_func, NUM_ACTIONS,
    player1, player2,
    game_indices, root_indices, root_moves,
    children, visit_counts, values, prior_probs,
    is_expanded, is_terminal, terminal_values,
    boards, players, parents, edge_from_parent,
    next_free_idx_ptr, max_nodes, depths, max_game_depth,
    c_visit=50.0, c_scale=1.0
):
    """
    Tree traversal for Gumbel MCTS (Full Gumbel variant).
    
    - Root move is forced by Sequential Halving
    - Non-root nodes use deterministic π'-based selection (Section 5, Equation 14)
    
    This minimizes approximation error by matching empirical visit counts
    to the improved policy π' = softmax(logits + σ(completedQ)).
    """
    n_active = len(game_indices)
    leaf_indices = np.zeros(n_active, dtype=np.int32)
    
    for i in range(n_active):
        node_idx = root_indices[i]
        search_depth = 0
        
        # Root move is forced by the Gumbel selection logic
        move_to_take = root_moves[i]
        
        while True:
            # Check termination conditions
            if is_terminal[node_idx] or not is_expanded[node_idx] or search_depth >= max_game_depth:
                break
            
            search_depth += 1
            
            # Non-root nodes: deterministic selection (Section 5)
            if search_depth > 1:
                valid_mask = get_valid_mask_func(boards[node_idx], players[node_idx])
                parent_n = visit_counts[node_idx]
                
                # Completion baseline: node's backed-up value
                v_node = values[node_idx] / parent_n if parent_n > 0 else 0.0
                
                # 1. Gather Q-values and statistics in one pass
                sum_child_n = 0
                max_child_n = 0
                node_q = np.zeros(NUM_ACTIONS, dtype=np.float64)
                
                for a in range(NUM_ACTIONS):
                    c_idx = children[node_idx, a]
                    if c_idx != -1:
                        n_c = visit_counts[c_idx]
                        sum_child_n += n_c
                        if n_c > 0:
                            # Child value negated (opponent's perspective -> my perspective)
                            node_q[a] = -values[c_idx] / n_c
                            if n_c > max_child_n:
                                max_child_n = n_c
                        else:
                            # Allocated but unvisited: use completion
                            node_q[a] = v_node 
                    else:
                        # Never allocated: use completion
                        node_q[a] = v_node
                
                # 2. Compute π' = softmax(logits + σ(completedQ)) [Equation 11]
                # Dynamic σ scale per Equation 8
                sigma_scale = (c_visit + max_child_n) * c_scale
                
                # Build combined scores
                combined = np.zeros(NUM_ACTIONS, dtype=np.float64)
                max_combined = -1e10
                
                q_min = 1e10
                q_max = -1e10
                for a in range(NUM_ACTIONS):
                    if valid_mask[a]:
                        if node_q[a] < q_min:
                            q_min = node_q[a]
                        if node_q[a] > q_max:
                            q_max = node_q[a]

                q_range = q_max - q_min
                if q_range < 1e-6:
                    q_range = 1.0

                for a in range(NUM_ACTIONS):
                    if valid_mask[a]:
                        log_prior = np.log(prior_probs[node_idx, a] + 1e-10)
                        q_normalized = (node_q[a] - q_min) / q_range
                        combined[a] = log_prior + sigma_scale * q_normalized
                        if combined[a] > max_combined:
                            max_combined = combined[a]
                    else:
                        combined[a] = -1e10
                
                # Stable softmax to get π'
                pi_prime = np.zeros(NUM_ACTIONS, dtype=np.float64)
                exp_sum = 0.0
                
                for a in range(NUM_ACTIONS):
                    if valid_mask[a]:
                        pi_prime[a] = np.exp(combined[a] - max_combined)
                        exp_sum += pi_prime[a]
                
                if exp_sum > 0:
                    for a in range(NUM_ACTIONS):
                        pi_prime[a] /= exp_sum
                
                # 3. Deterministic selection [Equation 14]
                # arg max_a (π'(a) - N(a) / (1 + Σ_b N(b)))
                denom = 1.0 + sum_child_n
                best_score = -1e10
                move_to_take = -1
                
                for a in range(NUM_ACTIONS):
                    if not valid_mask[a]:
                        continue
                    
                    # N(a) is the visit count of the specific child
                    c_idx = children[node_idx, a]
                    n_a = visit_counts[c_idx] if c_idx != -1 else 0
                    
                    # Equation 14: selects actions proportionally to π' over time
                    score = pi_prime[a] - (n_a / denom)
                    
                    if score > best_score:
                        best_score = score
                        move_to_take = a
            
            # 4. Transition to child node
            child_idx = children[node_idx, move_to_take]
            
            if child_idx == -1:
                # Allocate new node
                new_idx = next_free_idx_ptr[0]
                if new_idx >= max_nodes:
                    leaf_indices[i] = node_idx
                    break
                next_free_idx_ptr[0] += 1
                
                # Simulate the move
                board_copy = boards[node_idx].copy()
                curr_player = players[node_idx]
                reward, _, done, _ = fast_step_func(board_copy, move_to_take, curr_player)
                
                # Terminal value from child's perspective
                t_val = -1.0 if reward == 1.0 else (1.0 if reward == -1.0 else 0.0)
                next_p = player2 if curr_player == player1 else player1
                
                # Initialize the new node
                _init_node(
                    new_idx, node_idx, depths, move_to_take, board_copy, next_p,
                    children, parents, visit_counts, values, is_expanded, is_terminal,
                    terminal_values, boards, players, edge_from_parent, t_val, done
                )
                
                children[node_idx, move_to_take] = new_idx
                node_idx = new_idx
                break
            else:
                node_idx = child_idx
                
        leaf_indices[i] = node_idx
    
    return leaf_indices

@njit(cache=True)
def get_forced_root_moves_kernel(n_active, candidate_mask, forced_rank):
    """
    Picks the j-th (forced_rank) surviving candidate for each game.
    """
    selected_moves = np.zeros(n_active, dtype=np.int32)
    for i in range(n_active):
        # Find all legal candidates for this game
        active_moves = np.where(candidate_mask[i])[0]
        # Safety: if we have fewer candidates than forced_rank, wrap around
        # (Though in SH logic, we should have exactly k_phase candidates)
        if len(active_moves) == 0:
            selected_moves[i] = 0
            continue        
        idx = forced_rank % len(active_moves)
        selected_moves[i] = active_moves[idx]
    return selected_moves

@njit(cache=True)
def compute_gumbel_policy_kernel(
    n_active, root_indices, children, visit_counts, values,
    prior_probs, root_logits, legal_masks, nn_values,
    c_visit=50.0, c_scale=1.0
):
    num_actions = root_logits.shape[1]
    target_policies = np.zeros((n_active, num_actions), dtype=np.float32)
    
    for i in range(n_active):
        r_idx = root_indices[i]
        v_hat = nn_values[i]
        
        sum_n = visit_counts[r_idx] - 1
        sum_weighted_q = 0.0
        sum_pi_visited = 0.0
        max_n = 0
        
        q_values = np.zeros(num_actions, dtype=np.float32)
        
        for move in range(num_actions):
            c_idx = children[r_idx, move]
            if c_idx != -1:
                n_v = visit_counts[c_idx]
                if n_v > 0:
                    q_v = -values[c_idx] / n_v
                    q_values[move] = q_v
                    
                    pi_a = prior_probs[r_idx, move]
                    sum_weighted_q += pi_a * q_v
                    sum_pi_visited += pi_a
                    if n_v > max_n:
                        max_n = n_v
        
        # v_mix (Eq. 33)
        if sum_pi_visited > 1e-10 and sum_n > 0:
            v_mix = (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
        else:
            v_mix = v_hat
        
        # Complete Q-values and build combined scores
        sigma_scale = (c_visit + max_n) * c_scale
        
        q_min = 1e10
        q_max = -1e10
        for move in range(num_actions):
            if not legal_masks[i, move]:
                continue
            c_idx = children[r_idx, move]
            if c_idx != -1 and visit_counts[c_idx] > 0:
                q_completed = q_values[move]
            else:
                q_completed = v_mix
            q_values[move] = q_completed  # store completed value back
            if q_completed < q_min:
                q_min = q_completed
            if q_completed > q_max:
                q_max = q_completed

        q_range = q_max - q_min
        if q_range < 1e-6:
            q_range = 1.0
            
        max_combined = -1e10

        for move in range(num_actions):
            if not legal_masks[i, move]:
                continue
            q_normalized = (q_values[move] - q_min) / q_range
            combined = root_logits[i, move] + sigma_scale * q_normalized
            target_policies[i, move] = combined
            if combined > max_combined:
                max_combined = combined
        
        # Stable softmax
        exp_sum = 0.0
        for move in range(num_actions):
            if legal_masks[i, move]:
                target_policies[i, move] = np.exp(target_policies[i, move] - max_combined)
                exp_sum += target_policies[i, move]
            else:
                target_policies[i, move] = 0.0
        
        if exp_sum > 0:
            for move in range(num_actions):
                target_policies[i, move] /= exp_sum
    
    return target_policies