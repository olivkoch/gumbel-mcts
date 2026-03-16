import numpy as np
import torch
from kernels.puct_kernels import (
    select_leaves_batch, 
    backpropagate_batch, 
    _init_node
)


class PUCTStorage:
    def __init__(self, n_games, max_nodes, logic):

        self.num_actions = logic.NUM_ACTIONS
        self.board_shape = logic.BOARD_SHAPE
                    
        self.n_games = n_games
        
        # Structure (Int32 is fine for indices/counts)
        self.children = np.full((max_nodes, self.num_actions), -1, dtype=np.int32)
        self.parents = np.full(max_nodes, -1, dtype=np.int32)
        self.edge_from_parent = np.zeros(max_nodes, dtype=np.int16)
        self.visit_counts = np.zeros(max_nodes, dtype=np.int32)
        
        # === PRECISION FIX: Use float64 to match Python V2 ===
        self.values = np.zeros(max_nodes, dtype=np.float64)           # WAS float32
        self.prior_probs = np.zeros((max_nodes, self.num_actions), dtype=np.float64) # WAS float32
        self.terminal_values = np.zeros(max_nodes, dtype=np.float64)  # WAS float32
        
        self.is_expanded = np.zeros(max_nodes, dtype=np.bool_)
        self.is_terminal = np.zeros(max_nodes, dtype=np.bool_)
        
        self.boards = np.zeros((max_nodes, *self.board_shape), dtype=np.int8)
        self.players = np.zeros(max_nodes, dtype=np.int8)
        self.root_indices = np.zeros(n_games, dtype=np.int32)

        # track depth for monitoring
        self.depths = np.zeros(max_nodes, dtype=np.int32)

    def reset(self):
        # Optional: Zero out arrays if you reuse the object heavily
        pass

class PUCT:
    def __init__(self, n_games, max_nodes, logic, device='cuda'):
        
        self.logic = logic
        self.storage = PUCTStorage(n_games=n_games, max_nodes=max_nodes, logic=logic)       
        self.max_nodes = max_nodes
        self.device = device
        self.next_free_idx_arr = np.array([1], dtype=np.int32)
        
        # Pre-compute autocast settings
        if device == 'cuda' or (isinstance(device, str) and 'cuda' in device):
            self._autocast_device = 'cuda'
            self._use_autocast = True
            self._autocast_dtype = torch.bfloat16
        elif device == 'mps':
            self._autocast_device = 'mps'
            self._use_autocast = False  # MPS autocast is limited
            self._autocast_dtype = torch.float16
        else:
            self._autocast_device = 'cpu'
            self._use_autocast = False
            self._autocast_dtype = torch.float16
            
    def reset(self):
        num_used = self.next_free_idx_arr[0]  # capture BEFORE reset
        self.next_free_idx_arr[0] = 1
        self.storage.depths[:num_used + 1] = 0
        self.storage.is_expanded[:num_used + 1] = False
        self.storage.is_terminal[:num_used + 1] = False


    def initialize_roots(self, active_games, starting_boards, starting_players):
        current_alloc = self.next_free_idx_arr[0]
        for i, game_idx in enumerate(active_games):
            root_idx = current_alloc + i
            self.storage.root_indices[game_idx] = root_idx
            
            _init_node(
                idx=root_idx, parent=-1, depths=self.storage.depths, edge=-1, 
                board=starting_boards[i], player=starting_players[i],
                children=self.storage.children, parents=self.storage.parents,
                visit_counts=self.storage.visit_counts, values=self.storage.values,
                is_expanded=self.storage.is_expanded, is_terminal=self.storage.is_terminal,
                terminal_values=self.storage.terminal_values,
                boards=self.storage.boards, players=self.storage.players,
                edge_from_parent=self.storage.edge_from_parent,
                terminal_value=0.0, done_state=False
            )
        self.next_free_idx_arr[0] += len(active_games)

    def run_simulation_batch(self, model, active_games, num_simulations=50, 
                            c_puct_base=19652, c_puct_init=1.25):
        game_indices = np.array(active_games, dtype=np.int32)
        logic = self.logic
                
        # =========================================================================
        # PRE-EXPAND ROOTS (matches V2 semantics where root is expanded before loop)
        # =========================================================================
        unexpanded_root_indices = []
        for game_idx in active_games:
            root_idx = self.storage.root_indices[game_idx]
            if not self.storage.is_expanded[root_idx]:
                unexpanded_root_indices.append(root_idx)
        
        if unexpanded_root_indices:
            root_indices = np.array(unexpanded_root_indices, dtype=np.int32)
            
            leaves_boards = self.storage.boards[root_indices]
            leaves_players = self.storage.players[root_indices]
            
            obs_boards = torch.tensor(leaves_boards, device=self.device, dtype=torch.float32)
            obs_players = torch.tensor(leaves_players, device=self.device, dtype=torch.long)
            obs_boards_flat = obs_boards.flatten(1)
            
            batch = {"boards": obs_boards_flat, "current_player": obs_players}
            
            with torch.no_grad():
                outputs = model.forward_for_mcts(batch)
            
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            
            self.storage.is_expanded[root_indices] = True
            self.storage.prior_probs[root_indices] = priors
            
            backpropagate_batch(
                root_indices, vals,
                self.storage.parents, self.storage.visit_counts, self.storage.values
            )

        while True:
            # Check if all roots have enough visits
            min_visits = min(
                self.storage.visit_counts[self.storage.root_indices[g]] 
                for g in active_games
            )
            if min_visits >= num_simulations + 1:
                break

            # 1. Selection & Expansion
            leaf_indices = select_leaves_batch(
                fast_step_func=logic.fast_step,
                get_valid_mask_func=logic.get_valid_mask,
                NUM_ACTIONS=logic.NUM_ACTIONS,
                player1=logic.PLAYER_1,
                player2=logic.PLAYER_2,
                game_indices=game_indices, root_indices=self.storage.root_indices,
                children=self.storage.children, visit_counts=self.storage.visit_counts,
                values=self.storage.values, prior_probs=self.storage.prior_probs,
                is_expanded=self.storage.is_expanded, is_terminal=self.storage.is_terminal,
                terminal_values=self.storage.terminal_values,
                boards=self.storage.boards, players=self.storage.players,
                parents=self.storage.parents, edge_from_parent=self.storage.edge_from_parent,
                next_free_idx_ptr=self.next_free_idx_arr, c_puct_base=c_puct_base, c_puct_init=c_puct_init,
                max_nodes=self.max_nodes,
                depths=self.storage.depths,
                max_game_depth=logic.MAX_MOVES
            )
            
            # 2. Evaluation
            is_term = self.storage.is_terminal[leaf_indices]
            leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
            
            # A. Terminals - use ground truth value
            if np.any(is_term):
                leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
            
            # B. Non-Terminals - evaluate with NN
            non_term_mask = ~is_term
            if np.any(non_term_mask):
                nn_indices = leaf_indices[non_term_mask]
                
                leaves_boards = self.storage.boards[nn_indices]
                leaves_players = self.storage.players[nn_indices]
                
                obs_boards = torch.tensor(leaves_boards, device=self.device, dtype=torch.float32)
                obs_players = torch.tensor(leaves_players, device=self.device, dtype=torch.long)
                obs_boards_flat = obs_boards.flatten(1)
                
                batch = {"boards": obs_boards_flat, "current_player": obs_players}
                
                with torch.no_grad():
                    if self._use_autocast:
                        with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                            outputs = model.forward_for_mcts(batch)
                    else:
                        outputs = model.forward_for_mcts(batch)
                
                priors = outputs['policy'].float().cpu().numpy()
                vals = outputs['value'].float().cpu().numpy().flatten()
                
                self.storage.is_expanded[nn_indices] = True
                self.storage.prior_probs[nn_indices] = priors.astype(np.float64)
                leaf_values[non_term_mask] = vals.astype(np.float64)
            
            # 3. Single backprop for all leaves (exactly 1 per iteration)
            backpropagate_batch(
                leaf_indices, leaf_values,
                self.storage.parents, self.storage.visit_counts, self.storage.values
            )
    
    def get_root_data(self, game_idx):
        root_idx = self.storage.root_indices[game_idx]
        visit_counts = self.storage.visit_counts
        num_actions = self.storage.num_actions

        # Return as float32 for outside consumption (optional, but standard for policies)
        child_visits = np.zeros(num_actions, dtype=np.float32)
        for move in range(num_actions):
            child_idx = self.storage.children[root_idx, move]
            if child_idx != -1:
                child_visits[move] = visit_counts[child_idx]
                
        root_N = visit_counts[root_idx]
        root_W = self.storage.values[root_idx]
        root_Q = root_W / root_N if root_N > 0 else 0.0
        
        return child_visits, root_Q
    
    def get_all_root_data(self, n_active):
        """
        Vectorized extraction of visit counts and Q values for all active games.
        
        Args:
            n_active: Number of active games (uses indices 0 to n_active-1)
            
        Returns:
            child_visits: (n_active, NUM_ACTIONS) visit counts per child
            root_Q: (n_active,) Q values at root
        """
        root_indices = self.storage.root_indices[:n_active]  # (n_active,)
        
        # Get children for all roots: (n_active, NUM_ACTIONS)
        children = self.storage.children[root_indices]
        
        # Mask for valid children
        valid_mask = children != -1
        
        # Replace -1 with 0 for safe indexing (results will be masked anyway)
        safe_children = np.where(valid_mask, children, 0)
        
        # Vectorized visit count lookup
        child_visits = self.storage.visit_counts[safe_children].astype(np.float32)
        child_visits = np.where(valid_mask, child_visits, 0)
        
        # Vectorized Q calculation
        root_N = self.storage.visit_counts[root_indices]
        root_W = self.storage.values[root_indices]
        root_Q = np.divide(root_W, root_N, out=np.zeros_like(root_W), where=root_N > 0)
        
        return child_visits, root_Q.astype(np.float32)
        
    def get_max_depth(self):
        """Returns the absolute peak depth found in the storage."""
        # next_free_idx_arr[0] is the total number of nodes allocated
        upper_bound = self.next_free_idx_arr[0]
        if upper_bound <= 1:
            return 0
            
        # peak_d is simply the max value in the used portion of the depth array
        return np.max(self.storage.depths[:upper_bound])
    
    def get_all_root_visits(self, n_active=None):
        """Return (n_active, NUM_ACTIONS) visit counts for all active roots."""
        if n_active is None:
            n_active = self.storage.n_games
        child_visits, _ = self.get_all_root_data(n_active)
        return child_visits