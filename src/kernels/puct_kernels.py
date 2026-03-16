import numpy as np
from numba import njit

# =============================================================================
# HELPER KERNELS
# =============================================================================

@njit(cache=True)
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
    
    # CRITICAL: Use explicit done_state for is_terminal (catches Draws where value=0.0)
    is_terminal[idx] = done_state
    terminal_values[idx] = terminal_value
    
    boards[idx] = board 
    players[idx] = player

    # used for monitoring only
    if parent == -1:
        depths[idx] = 0
    else:
        depths[idx] = depths[parent] + 1

@njit(cache=False)
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
    """
    SELECTION PHASE: Traverse from roots to find leaves for expansion.
    
    Args:
        fast_step_func: Function to execute a move on the board
        get_valid_mask_func: Function to get valid move mask
        NUM_ACTIONS: Number of possible actions in the game
        game_indices: Array of active game indices
        root_indices: Array mapping game_idx -> root node index
        children: (max_nodes, NUM_ACTIONS) child node indices, -1 if not created
        visit_counts: (max_nodes,) visit count per node
        values: (max_nodes,) cumulative value per node
        prior_probs: (max_nodes, NUM_ACTIONS) prior probabilities
        is_expanded: (max_nodes,) whether node has been expanded by NN
        is_terminal: (max_nodes,) whether node is terminal (game over)
        terminal_values: (max_nodes,) ground truth value for terminal nodes
        boards: (max_nodes, 6, 7) board state per node
        players: (max_nodes,) current player per node
        parents: (max_nodes,) parent node index, -1 for roots
        edge_from_parent: (max_nodes,) action that led to this node
        next_free_idx_ptr: (1,) pointer to next free node index
        c_puct_base: PUCT exploration constant
        c_puct_init: PUCT exploration constant
        max_nodes: Maximum number of nodes allowed
        depths: node depths, used for monitoring only
        
    Returns:
        leaf_indices: Array of leaf node indices to evaluate
    """
    n_active = len(game_indices)
    leaf_indices = np.zeros(n_active, dtype=np.int32)
    
    # Explicit float casting for precision matching
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

            # Compute legal actions from board state
            valid_mask = get_valid_mask_func(boards[node_idx], players[node_idx])
            
            best_score = -1e9
            best_move = -1
            parent_N = visit_counts[node_idx]
            sqrt_parent_N = np.sqrt(parent_N)
            
            # Calculate PUCT constant (Float64)
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
                        child_Q = -values[child_idx] / child_N
                
                prior = prior_probs[node_idx, move]
                
                # UCB Score = Q + U
                u_score = pb_c * prior * sqrt_parent_N / (1.0 + child_N)
                score = child_Q + u_score
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if not has_valid_move:
                break
            
            child_idx = children[node_idx, best_move]
            
            if child_idx == -1:
                # ALLOCATE NEW CHILD
                new_idx = next_free_idx_ptr[0]
                if new_idx >= max_nodes:
                    leaf_indices[i] = node_idx
                    break
                next_free_idx_ptr[0] += 1
                
                current_board = boards[node_idx].copy()
                current_player = players[node_idx]
                
                reward, _, done, _ = fast_step_func(current_board, best_move, current_player)
                
                term_val = 0.0
                if done:
                    if reward == 1.0:
                        term_val = -1.0
                    elif reward == -1.0:
                        term_val = 1.0
                    else:
                        term_val = 0.0
                
                next_player = player2 if current_player == player1 else player1
                
                _init_node(
                    new_idx, node_idx, depths, best_move, current_board, next_player,
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

@njit(cache=False)
def backpropagate_batch(leaf_indices, nn_values, parents, visit_counts, values):
    n_leaves = len(leaf_indices)
    for i in range(n_leaves):
        node_idx = leaf_indices[i]
        value = nn_values[i]
        
        while node_idx != -1:
            visit_counts[node_idx] += 1
            values[node_idx] += value
            value = -value
            node_idx = parents[node_idx]

@njit(cache=False)
def sample_moves_batch(policies, warm_mask, rng_vals):
    """
    Sample moves from policies for warm games, argmax for cold games.
    
    Args:
        policies: (n_games, NUM_ACTIONS) normalized probabilities
        warm_mask: (n_games,) True if sampling, False if argmax
        rng_vals: (n_games,) random values in [0, 1) for sampling
        
    Returns:
        moves: (n_games,) selected actions
    """
    n_games, num_actions = policies.shape
    moves = np.zeros(n_games, dtype=np.int32)
    
    for i in range(n_games):
        if warm_mask[i]:
            # Sample from policy using inverse CDF
            cumsum = 0.0
            r = rng_vals[i]
            for a in range(num_actions):
                cumsum += policies[i, a]
                if r < cumsum:
                    moves[i] = a
                    break
            else:
                moves[i] = num_actions - 1  # Fallback
        else:
            # Argmax
            best = 0
            best_val = policies[i, 0]
            for a in range(1, num_actions):
                if policies[i, a] > best_val:
                    best_val = policies[i, a]
                    best = a
            moves[i] = best
    
    return moves