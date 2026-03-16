"""
Sparse Gumbel MCTS kernels (v5).

Replaces dense (max_nodes, NUM_ACTIONS) arrays with a flat edge list,
enabling chess-scale action spaces (4672 actions) without blowing up memory.

Key data structures:
  - Per-node: node_edge_offset, node_num_edges  (where this node's edges start/how many)
  - Flat edge arrays: edge_action, edge_child, edge_prior
  - Per-node stats: visit_counts, values  (unchanged from v3/v4)

Interior node selection uses the Gumbel MuZero formula:
    a* = argmax_a [ sigma(s) * q_normalized(s,a) + log_prior(a) ]
  where sigma(s) = c_scale * (c_visit + N_parent)
"""

import numpy as np
from numba import njit

# =============================================================================
# TREE DESCENT (hot path — called once per simulation per candidate)
# =============================================================================

@njit(cache=True)
def _compute_v_mix(node_idx, edge_start, n_edges, edge_child, edge_prior,
                   visit_counts, values, node_nn_value):
    """
    Compute mixed value v_mix(s) for a node:
      v_mix = (sum_N * Q_bar + v_nn) / (sum_N + 1)
    where Q_bar = weighted average Q across visited children.
    Falls back to v_nn when no children have visits.
    """
    v_hat = node_nn_value[node_idx]
    sum_n = visit_counts[node_idx] - 1
    if sum_n < 0:
        sum_n = 0
    sum_weighted_q = 0.0
    sum_pi_visited = 0.0

    for e in range(n_edges):
        eidx = edge_start + e
        c_idx = edge_child[eidx]
        if c_idx != -1:
            n_c = visit_counts[c_idx]
            if n_c > 0:
                q_c = -values[c_idx] / n_c  # negated: opponent perspective
                pi_a = edge_prior[eidx]
                sum_weighted_q += pi_a * q_c
                sum_pi_visited += pi_a

    if sum_pi_visited > 1e-10 and sum_n > 0:
        return (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
    return v_hat


@njit(cache=True)
def _select_edge(node_idx, edge_start, n_edges,
                 edge_child, edge_prior,
                 visit_counts, values, node_nn_value,
                 c_visit, c_scale):
    """
    Select the best edge at an interior node using sigma-scaled completed Q.
    Returns the LOCAL edge index (0..n_edges-1).

        Formula:
            score(a) = sigma * q_normalized(a) + log(prior(a))
      sigma = c_scale * (c_visit + N_parent)
      q_completed(a) = Q(a) if visited, else v_mix(s)
    """
    v_mix = _compute_v_mix(node_idx, edge_start, n_edges, edge_child, edge_prior,
                           visit_counts, values, node_nn_value)

    q_values = np.empty(n_edges, dtype=np.float64)
    q_min = 1e10
    q_max = -1e10
    max_child_n = 0

    for e in range(n_edges):
        eidx = edge_start + e
        c_idx = edge_child[eidx]
        n_c = 0

        if c_idx != -1 and visit_counts[c_idx] > 0:
            n_c = visit_counts[c_idx]
            q = -values[c_idx] / n_c
        else:
            q = v_mix

        if n_c > max_child_n:
            max_child_n = n_c

        q_values[e] = q
        if q < q_min:
            q_min = q
        if q > q_max:
            q_max = q

    q_range = q_max - q_min
    if q_range < 1e-6:
        q_range = 1.0

    sigma = c_scale * (c_visit + max_child_n)

    combined = np.empty(n_edges, dtype=np.float64)
    max_combined = -1e18
    for e in range(n_edges):
        eidx = edge_start + e
        p = edge_prior[eidx]
        log_p = np.log(max(p, 1e-10))
        q_normalized = (q_values[e] - q_min) / q_range
        combined[e] = log_p + sigma * q_normalized
        if combined[e] > max_combined:
            max_combined = combined[e]

    pi_prime = np.empty(n_edges, dtype=np.float64)
    exp_sum = 0.0
    for e in range(n_edges):
        pi_prime[e] = np.exp(combined[e] - max_combined)
        exp_sum += pi_prime[e]

    if exp_sum > 0.0:
        for e in range(n_edges):
            pi_prime[e] /= exp_sum

    sum_child_n = 0
    n_per_edge = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        eidx = edge_start + e
        c_idx = edge_child[eidx]
        n_a = visit_counts[c_idx] if c_idx != -1 else 0
        n_per_edge[e] = n_a
        sum_child_n += n_a

    denom = 1.0 + sum_child_n
    best_local = 0
    best_score = -1e18
    for e in range(n_edges):
        score = pi_prime[e] - (n_per_edge[e] / denom)
        if score > best_score:
            best_score = score
            best_local = e

    return best_local


@njit(cache=True)
def descend_batch(
    fast_step_func,
    game_indices,          # (n_active,) int32
    root_indices,          # (n_games,) int32 — maps game_idx -> root node
    forced_edge_locals,    # (n_active,) int32 — local edge idx at root, -1 if none
    # --- Node arrays ---
    visit_counts, values, is_expanded, is_terminal, terminal_values,
    boards, players, parents, edge_from_parent, depths, node_nn_value,
    # --- Edge arrays ---
    node_edge_offset, node_num_edges,
    edge_action, edge_child, edge_prior,
    # --- Allocator ---
    next_free_node_arr,    # (1,) int32
    max_nodes,
    # --- Params ---
    c_visit, c_scale, max_game_depth,
    player1, player2,
    board_rows, board_cols,
):
    """
    Descend the tree for each active game, returning one leaf per game.
    Creates child nodes on demand via fast_step.
    
    For the root move: uses forced_edge_locals (from Gumbel sequential halving).
    For interior nodes: uses sigma-scaled completed-Q selection.
    
    Returns:
        leaf_indices: (n_active,) int32 — node index of each leaf
    """
    n_active = len(game_indices)
    leaf_indices = np.empty(n_active, dtype=np.int32)

    for i in range(n_active):
        node_idx = root_indices[game_indices[i]]
        forced = forced_edge_locals[i]
        first_move = True

        while True:
            # Leaf or terminal → stop
            if is_terminal[node_idx]:
                leaf_indices[i] = node_idx
                break
            if not is_expanded[node_idx]:
                leaf_indices[i] = node_idx
                break

            e_start = node_edge_offset[node_idx]
            n_edges = node_num_edges[node_idx]

            if n_edges == 0:
                # Shouldn't happen for expanded non-terminal, but safety
                leaf_indices[i] = node_idx
                break

            # --- Pick edge ---
            if first_move and forced >= 0:
                chosen_local = forced
                first_move = False
            else:
                chosen_local = _select_edge(
                    node_idx, e_start, n_edges,
                    edge_child, edge_prior,
                    visit_counts, values, node_nn_value,
                    c_visit, c_scale[i]
                )

            eidx = e_start + chosen_local
            action = edge_action[eidx]
            child_idx = edge_child[eidx]

            # --- Create child node if it doesn't exist ---
            if child_idx == -1:
                new_idx = next_free_node_arr[0]
                if new_idx >= max_nodes:
                    # Out of node space → treat current as leaf
                    leaf_indices[i] = node_idx
                    break

                next_free_node_arr[0] = new_idx + 1
                edge_child[eidx] = new_idx

                # Init child via fast_step
                cur_board = boards[node_idx].copy().reshape(board_rows, board_cols)
                cur_player = players[node_idx]
                reward, winner, done, new_board = fast_step_func(
                    cur_board, action, cur_player
                )

                new_player = player2 if cur_player == player1 else player1

                boards[new_idx] = new_board.ravel()
                players[new_idx] = new_player
                parents[new_idx] = node_idx
                edge_from_parent[new_idx] = action
                depths[new_idx] = depths[node_idx] + 1
                visit_counts[new_idx] = 0
                values[new_idx] = 0.0
                is_expanded[new_idx] = False

                if done:
                    is_terminal[new_idx] = True
                    # reward is from perspective of player who just moved (cur_player)
                    # store from child's perspective (new_player), so negate
                    terminal_values[new_idx] = -reward
                else:
                    is_terminal[new_idx] = False
                    terminal_values[new_idx] = 0.0

                leaf_indices[i] = new_idx
                break

            # --- Child exists, continue descent ---
            node_idx = child_idx

    return leaf_indices

def descend_batch_python(
    fast_step_func,
    game_indices, root_indices, forced_edge_locals,
    visit_counts, values, is_expanded, is_terminal, terminal_values,
    boards, players, parents, edge_from_parent, depths, node_nn_value,
    node_edge_offset, node_num_edges,
    edge_action, edge_child, edge_prior,
    next_free_node_arr, max_nodes,
    c_visit, c_scale, max_game_depth,
    player1, player2,
    board_rows=0, board_cols=0,
):
    """
    Pure-Python descent for non-Numba game logics (e.g., python-chess).
    Same algorithm as descend_batch but calls fast_step_func as a regular Python function.
    Uses .copy() on boards to prevent in-place mutation of parent state.
    """
    n_active = len(game_indices)
    leaf_indices = np.empty(n_active, dtype=np.int32)

    for i in range(n_active):
        node_idx = root_indices[game_indices[i]]
        forced = int(forced_edge_locals[i])
        first_move = True

        while True:
            if is_terminal[node_idx]:
                leaf_indices[i] = node_idx
                break
            if not is_expanded[node_idx]:
                leaf_indices[i] = node_idx
                break

            e_start = int(node_edge_offset[node_idx])
            n_edges = int(node_num_edges[node_idx])

            if n_edges == 0:
                leaf_indices[i] = node_idx
                break

            if first_move and forced >= 0:
                chosen_local = forced
                first_move = False
            else:
                chosen_local = _select_edge(
                    node_idx, e_start, n_edges,
                    edge_child, edge_prior,
                    visit_counts, values, node_nn_value,
                    c_visit, c_scale[i]
                )

            eidx = e_start + chosen_local
            action = int(edge_action[eidx])
            child_idx = edge_child[eidx]

            if child_idx == -1:
                new_idx = next_free_node_arr[0]
                if new_idx >= max_nodes:
                    leaf_indices[i] = node_idx
                    break

                next_free_node_arr[0] = new_idx + 1
                edge_child[eidx] = new_idx

                cur_board = boards[node_idx].copy().reshape(board_rows, board_cols)
                cur_player = int(players[node_idx])
                reward, winner, done, new_board = fast_step_func(
                    cur_board, action, cur_player
                )
                new_player = player2 if cur_player == player1 else player1

                boards[new_idx] = new_board.ravel()
                players[new_idx] = new_player
                parents[new_idx] = node_idx
                edge_from_parent[new_idx] = action
                depths[new_idx] = depths[node_idx] + 1
                visit_counts[new_idx] = 0
                values[new_idx] = 0.0
                is_expanded[new_idx] = False

                if done:
                    is_terminal[new_idx] = True
                    terminal_values[new_idx] = -reward
                else:
                    is_terminal[new_idx] = False
                    terminal_values[new_idx] = 0.0

                leaf_indices[i] = new_idx
                break

            node_idx = child_idx

    return leaf_indices

# =============================================================================
# GUMBEL SCORING (called a few times per search — Python/NumPy is fine)
# =============================================================================

def get_forced_edge_local(candidate_mask, candidate_rank):
    """
    For each game, find the candidate_rank-th active candidate.
    candidate_mask: (n_active, max_legal) bool
    Returns: (n_active,) int32 — local edge index, or -1 if not found
    """
    n_active = candidate_mask.shape[0]
    edges = np.full(n_active, -1, dtype=np.int32)
    for i in range(n_active):
        count = 0
        for j in range(candidate_mask.shape[1]):
            if candidate_mask[i, j]:
                if count == candidate_rank:
                    edges[i] = j
                    break
                count += 1
    return edges


def compute_gumbel_scores(
    n_active, root_indices,
    node_edge_offset, node_num_edges,
    edge_child, edge_action, edge_prior,
    visit_counts, values, node_nn_value,
    root_logits,        # (n_active, max_legal) — log-priors for root legal moves
    root_gumbel_noise,  # (n_active, max_legal)
    root_num_legal,     # (n_active,) int16
    candidate_mask,     # (n_active, max_legal) bool
    c_visit, c_scale_vector,
):
    """
    Compute score = sigma * q_normalized(a) + logit(a) + gumbel(a)
    for all candidate actions at the root. Used for sequential halving.
    
    Returns: (n_active, max_legal) float32 — scores, -inf for non-candidates
    """
    max_legal = candidate_mask.shape[1]
    scores = np.full((n_active, max_legal), -1e18, dtype=np.float32)

    for i in range(n_active):
        r_idx = root_indices[i]
        e_start = node_edge_offset[r_idx]
        n_edges = node_num_edges[r_idx]

        v_mix = _compute_v_mix_py(
            r_idx, e_start, n_edges, edge_child, edge_prior,
            visit_counts, values, node_nn_value
        )

        n_legal = min(root_num_legal[i], n_edges)
        if n_legal <= 0:
            continue

        q_values = np.empty(n_legal, dtype=np.float64)
        q_min = 1e10
        q_max = -1e10
        max_child_n = 0

        for e in range(n_legal):
            eidx = e_start + e
            c_idx = edge_child[eidx]
            n_c = 0

            if c_idx != -1 and visit_counts[c_idx] > 0:
                n_c = visit_counts[c_idx]
                q = -values[c_idx] / n_c
            else:
                q = v_mix

            if n_c > max_child_n:
                max_child_n = n_c

            q_values[e] = q
            if q < q_min:
                q_min = q
            if q > q_max:
                q_max = q

        q_range = q_max - q_min
        if q_range < 1e-6:
            q_range = 1.0

        sigma = c_scale_vector[i] * (c_visit + max_child_n)

        for e in range(n_legal):
            if not candidate_mask[i, e]:
                continue

            scores[i, e] = (
                sigma * ((q_values[e] - q_min) / q_range)
                + root_logits[i, e]
                + root_gumbel_noise[i, e]
            )

    return scores


def compute_improved_policy(
    n_active, root_indices,
    node_edge_offset, node_num_edges,
    edge_child, edge_action, edge_prior,
    visit_counts, values, node_nn_value,
    root_logits,      # (n_active, max_legal)
    root_num_legal,   # (n_active,)
    c_visit, c_scale_vector,
    num_actions,      # full action space size (e.g. 4672 for chess)
):
    """
    Compute improved policy: pi'(a) = softmax( logit(a) + sigma * q_normalized(a) )
    Returns DENSE (n_active, num_actions) for training targets.
    """
    dense_policy = np.zeros((n_active, num_actions), dtype=np.float32)

    for i in range(n_active):
        r_idx = root_indices[i]
        e_start = node_edge_offset[r_idx]
        n_edges = node_num_edges[r_idx]

        v_mix = _compute_v_mix_py(
            r_idx, e_start, n_edges, edge_child, edge_prior,
            visit_counts, values, node_nn_value
        )

        n_legal = min(root_num_legal[i], n_edges)
        if n_legal <= 0:
            continue

        local_actions = np.empty(n_legal, dtype=np.int32)
        local_q = np.empty(n_legal, dtype=np.float64)
        q_min = 1e10
        q_max = -1e10
        max_child_n = 0

        for e in range(n_legal):
            eidx = e_start + e
            c_idx = edge_child[eidx]
            n_c = 0

            if c_idx != -1 and visit_counts[c_idx] > 0:
                n_c = visit_counts[c_idx]
                q = -values[c_idx] / n_c
            else:
                q = v_mix

            if n_c > max_child_n:
                max_child_n = n_c

            local_actions[e] = edge_action[eidx]
            local_q[e] = q
            if q < q_min:
                q_min = q
            if q > q_max:
                q_max = q

        q_range = q_max - q_min
        if q_range < 1e-6:
            q_range = 1.0

        sigma = c_scale_vector[i] * (c_visit + max_child_n)

        local_scores = np.empty(n_legal, dtype=np.float64)
        for e in range(n_legal):
            q_normalized = (local_q[e] - q_min) / q_range
            local_scores[e] = root_logits[i, e] + sigma * q_normalized

        # Stable softmax over legal moves only
        if n_legal > 0:
            local_scores -= local_scores.max()
            exp_scores = np.exp(local_scores)
            probs = exp_scores / max(exp_scores.sum(), 1e-10)
            for e in range(n_legal):
                dense_policy[i, local_actions[e]] = probs[e]

    return dense_policy


# =============================================================================
# Pure-Python v_mix (mirrors the Numba version for use outside @njit)
# =============================================================================

def _compute_v_mix_py(node_idx, edge_start, n_edges, edge_child, edge_prior,
                      visit_counts, values, node_nn_value):
    v_hat = node_nn_value[node_idx]
    sum_n = visit_counts[node_idx] - 1
    if sum_n < 0:
        sum_n = 0
    sum_weighted_q = 0.0
    sum_pi_visited = 0.0

    for e in range(n_edges):
        eidx = edge_start + e
        c_idx = edge_child[eidx]
        if c_idx != -1:
            n_c = visit_counts[c_idx]
            if n_c > 0:
                q_c = -values[c_idx] / n_c
                pi_a = edge_prior[eidx]
                sum_weighted_q += pi_a * q_c
                sum_pi_visited += pi_a

    if sum_pi_visited > 1e-10 and sum_n > 0:
        return (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
    return float(v_hat)