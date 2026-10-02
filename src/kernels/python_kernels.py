"""Pure-Python single-player MCTS kernels.

Use these for single-player environments whose fast_step cannot be
numba-compiled (e.g. PushT with pymunk physics).

Differs from the library's @njit kernels in two ways:
  1. No @njit — runs as regular Python (pymunk can't be compiled)
  2. No negamax — values are NOT negated during backpropagation
     or Q-value computation (single-player, no adversary)
"""

import numpy as np


# =============================================================================
# From puct_kernels.py
# =============================================================================

def _init_node(idx, parent, depths, edge, board, player, children, parents,
               visit_counts, values, is_expanded, is_terminal, terminal_values,
               boards, players, edge_from_parent, terminal_value, done_state):
    """Initialize a single node's data."""
    children[idx, :] = -1
    parents[idx] = parent
    edge_from_parent[idx] = edge

    visit_counts[idx] = 0
    values[idx] = 0.0
    is_expanded[idx] = False

    is_terminal[idx] = done_state
    terminal_values[idx] = terminal_value

    boards[idx] = board
    players[idx] = player

    if parent == -1:
        depths[idx] = 0
    else:
        depths[idx] = depths[parent] + 1


def select_leaves_batch(
    fast_step_func, get_valid_mask_func, NUM_ACTIONS,
    player1, player2,
    game_indices, root_indices,
    children, visit_counts, values, prior_probs,
    is_expanded, is_terminal, terminal_values,
    boards, players, parents, edge_from_parent,
    next_free_idx_ptr,
    c_puct_base, c_puct_init,
    max_nodes, depths, max_game_depth
):
    n_active = len(game_indices)
    leaf_indices = np.zeros(n_active, dtype=np.int32)

    f_c_puct_base = float(c_puct_base)
    f_c_puct_init = float(c_puct_init)

    for i in range(n_active):
        game_idx = game_indices[i]
        node_idx = root_indices[game_idx]

        search_depth = 0

        while True:
            if is_terminal[node_idx] or not is_expanded[node_idx] or search_depth >= max_game_depth:
                break

            search_depth += 1

            valid_mask = get_valid_mask_func(boards[node_idx], players[node_idx])

            best_score = -1e9
            best_move = -1
            parent_N = visit_counts[node_idx]
            sqrt_parent_N = np.sqrt(parent_N)

            pb_c = np.log((1.0 + parent_N + f_c_puct_base) / f_c_puct_base) + f_c_puct_init

            has_valid_move = False

            for move in range(NUM_ACTIONS):
                if not valid_mask[move]:
                    continue

                has_valid_move = True

                child_idx = children[node_idx, move]
                child_Q = 0.0
                child_N = 0

                if child_idx != -1:
                    child_N = visit_counts[child_idx]
                    if child_N > 0:
                        child_Q = values[child_idx] / child_N

                prior = prior_probs[node_idx, move]

                u_score = pb_c * prior * sqrt_parent_N / (1.0 + child_N)
                score = child_Q + u_score

                if score > best_score:
                    best_score = score
                    best_move = move

            if not has_valid_move:
                break

            child_idx = children[node_idx, best_move]

            if child_idx == -1:
                new_idx = next_free_idx_ptr[0]
                if new_idx >= max_nodes:
                    leaf_indices[i] = node_idx
                    break
                next_free_idx_ptr[0] += 1

                current_board = boards[node_idx].copy()
                current_player = players[node_idx]

                reward, _, done, _ = fast_step_func(current_board, best_move, current_player)

                term_val = reward if done else 0.0

                _init_node(
                    new_idx, node_idx, depths, best_move, current_board, player1,
                    children, parents, visit_counts, values, is_expanded, is_terminal,
                    terminal_values, boards, players, edge_from_parent,
                    term_val, done
                )

                children[node_idx, best_move] = new_idx
                node_idx = new_idx
                break
            else:
                node_idx = child_idx

        leaf_indices[i] = node_idx

    return leaf_indices


def backpropagate_batch(leaf_indices, nn_values, parents, visit_counts, values):
    n_leaves = len(leaf_indices)
    for i in range(n_leaves):
        node_idx = leaf_indices[i]
        value = nn_values[i]

        while node_idx != -1:
            visit_counts[node_idx] += 1
            values[node_idx] += value
            node_idx = parents[node_idx]


# =============================================================================
# From gumbel_dense_kernels.py
# =============================================================================

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

        v_hat = nn_values[i]
        sum_n = visit_counts[r_idx] - 1
        sum_weighted_q = 0.0
        sum_pi_visited = 0.0

        for m in range(num_actions):
            c_idx = children[r_idx, m]
            if c_idx != -1:
                n_c = visit_counts[c_idx]
                if n_c > max_n:
                    max_n = n_c
                if n_c > 0:
                    q_c = values[c_idx] / n_c
                    q_min = min(q_min, q_c)
                    q_max = max(q_max, q_c)

                    pi_a = prior_probs[r_idx, m]
                    sum_weighted_q += pi_a * q_c
                    sum_pi_visited += pi_a

        if sum_pi_visited > 1e-10 and sum_n > 0:
            v_mix = (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
        else:
            v_mix = v_hat

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
            q_v = (values[child_idx] / n_v) if n_v > 0 else v_mix

            q_normalized = (q_v - q_min) / q_range

            scores[i, move] = (root_logits[g_idx, move] +
                  gumbel_noises[g_idx, move] +
                  dynamic_q_scale * q_normalized)

    return scores


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
    n_active = len(game_indices)
    leaf_indices = np.zeros(n_active, dtype=np.int32)

    for i in range(n_active):
        node_idx = root_indices[i]
        search_depth = 0

        move_to_take = root_moves[i]

        while True:
            if is_terminal[node_idx] or not is_expanded[node_idx] or search_depth >= max_game_depth:
                break

            search_depth += 1

            if search_depth > 1:
                valid_mask = get_valid_mask_func(boards[node_idx], players[node_idx])
                parent_n = visit_counts[node_idx]

                v_node = values[node_idx] / parent_n if parent_n > 0 else 0.0

                sum_child_n = 0
                max_child_n = 0
                node_q = np.zeros(NUM_ACTIONS, dtype=np.float64)

                for a in range(NUM_ACTIONS):
                    c_idx = children[node_idx, a]
                    if c_idx != -1:
                        n_c = visit_counts[c_idx]
                        sum_child_n += n_c
                        if n_c > 0:
                            node_q[a] = values[c_idx] / n_c
                            if n_c > max_child_n:
                                max_child_n = n_c
                        else:
                            node_q[a] = v_node
                    else:
                        node_q[a] = v_node

                sigma_scale = (c_visit + max_child_n) * c_scale

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

                pi_prime = np.zeros(NUM_ACTIONS, dtype=np.float64)
                exp_sum = 0.0

                for a in range(NUM_ACTIONS):
                    if valid_mask[a]:
                        pi_prime[a] = np.exp(combined[a] - max_combined)
                        exp_sum += pi_prime[a]

                if exp_sum > 0:
                    for a in range(NUM_ACTIONS):
                        pi_prime[a] /= exp_sum

                denom = 1.0 + sum_child_n
                best_score = -1e10
                move_to_take = -1

                for a in range(NUM_ACTIONS):
                    if not valid_mask[a]:
                        continue

                    c_idx = children[node_idx, a]
                    n_a = visit_counts[c_idx] if c_idx != -1 else 0

                    score = pi_prime[a] - (n_a / denom)

                    if score > best_score:
                        best_score = score
                        move_to_take = a

            child_idx = children[node_idx, move_to_take]

            if child_idx == -1:
                new_idx = next_free_idx_ptr[0]
                if new_idx >= max_nodes:
                    leaf_indices[i] = node_idx
                    break
                next_free_idx_ptr[0] += 1

                board_copy = boards[node_idx].copy()
                curr_player = players[node_idx]
                reward, _, done, _ = fast_step_func(board_copy, move_to_take, curr_player)

                t_val = reward if done else 0.0

                _init_node(
                    new_idx, node_idx, depths, move_to_take, board_copy, player1,
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


def get_forced_root_moves_kernel(n_active, candidate_mask, forced_rank):
    selected_moves = np.zeros(n_active, dtype=np.int32)
    for i in range(n_active):
        active_moves = np.where(candidate_mask[i])[0]
        if len(active_moves) == 0:
            selected_moves[i] = 0
            continue
        idx = forced_rank % len(active_moves)
        selected_moves[i] = active_moves[idx]
    return selected_moves


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
                    q_v = values[c_idx] / n_v
                    q_values[move] = q_v

                    pi_a = prior_probs[r_idx, move]
                    sum_weighted_q += pi_a * q_v
                    sum_pi_visited += pi_a
                    if n_v > max_n:
                        max_n = n_v

        if sum_pi_visited > 1e-10 and sum_n > 0:
            v_mix = (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
        else:
            v_mix = v_hat

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
            q_values[move] = q_completed
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
