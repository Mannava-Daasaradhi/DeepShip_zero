import torch
import numpy as np
from src.agents.probability import ProbabilityAgent
from src.model.net import DeepShipNet

class AlphaZeroAgent:
    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DeepShipNet().to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except:
                print("Could not load weights. Starting fresh.")
        
        self.model.eval() 
        self.math_agent = ProbabilityAgent(device=self.device)
        
        # Parity Mask (Checkerboard)
        self.parity_mask = np.zeros((10, 10), dtype=np.float32)
        for r in range(10):
            for c in range(10):
                if (r + c) % 2 == 0:
                    self.parity_mask[r, c] = 1.0

    def preprocess_state(self, board_state, sunk_ships):
        """Converts the board into a (1, 3, 10, 10) tensor."""
        hits = (board_state == 1).astype(np.float32)
        misses = (board_state == 2).astype(np.float32)
        heatmap = self.math_agent.generate_heatmap(board_state, sunk_ships)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        state_stack = np.stack([hits, misses, heatmap])
        return torch.tensor(state_stack).unsqueeze(0).to(self.device)

    def get_action(self, board_state, sunk_ships, epsilon=0.0):
        """Standard fast action (Fallback)."""
        valid_moves = (board_state == 0).astype(np.float32)
        if np.random.random() < epsilon:
            candidates = np.argwhere(valid_moves)
            return tuple(candidates[np.random.choice(len(candidates))])

        state_tensor = self.preprocess_state(board_state, sunk_ships)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().squeeze()
        policy_probs = policy_probs * valid_moves.flatten()
        
        if np.sum(policy_probs) > 0:
            policy_probs /= np.sum(policy_probs)
            best_move_flat = np.argmax(policy_probs)
            return np.unravel_index(best_move_flat, (10, 10))
        else:
            return self.math_agent.get_action(board_state, sunk_ships)

    def get_action_mcts(self, board_state, sunk_ships, top_k=8):
        """
        SINGULARITY MODE: Dynamic Depth Search.
        """
        
        # --- PHASE 1: TERMINATOR (Kill Wounded) ---
        wounded_coords = np.argwhere(board_state == 1)
        if len(wounded_coords) > 0:
            targets = []
            for r, c in wounded_coords:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for nr, nc in neighbors:
                    if 0 <= nr < 10 and 0 <= nc < 10 and board_state[nr, nc] == 0:
                        targets.append((nr, nc))
            
            if len(wounded_coords) >= 2:
                rows, cols = zip(*wounded_coords)
                is_horizontal = len(set(rows)) == 1
                is_vertical = len(set(cols)) == 1
                
                better_targets = []
                if is_horizontal:
                    r = rows[0]
                    if 0 <= min(cols) - 1 < 10 and board_state[r, min(cols) - 1] == 0: better_targets.append((r, min(cols) - 1))
                    if 0 <= max(cols) + 1 < 10 and board_state[r, max(cols) + 1] == 0: better_targets.append((r, max(cols) + 1))
                elif is_vertical:
                    c = cols[0]
                    if 0 <= min(rows) - 1 < 10 and board_state[min(rows) - 1, c] == 0: better_targets.append((min(rows) - 1, c))
                    if 0 <= max(rows) + 1 < 10 and board_state[max(rows) + 1, c] == 0: better_targets.append((max(rows) + 1, c))
                
                if better_targets: targets = better_targets

            if targets:
                # Use Math to break ties
                current_probs = self.math_agent.generate_heatmap(board_state, sunk_ships)
                best_target = targets[0]
                best_prob = -1
                for r, c in targets:
                    if current_probs[r, c] > best_prob:
                        best_prob = current_probs[r, c]
                        best_target = (r, c)
                return best_target


        # --- PHASE 2: HUNT MODE (Adaptive Depth) ---
        
        # 1. Generate Probabilities
        current_probs = self.math_agent.generate_heatmap(board_state, sunk_ships)
        mask = (board_state == 0)
        
        # 2. Apply Parity Filter (Strictly enforce Checkerboard)
        current_probs = current_probs * mask * self.parity_mask
        
        if np.sum(current_probs) == 0:
             current_probs = self.math_agent.generate_heatmap(board_state, sunk_ships) * mask

        if np.sum(current_probs) == 0:
            return self.get_action(board_state, sunk_ships) 

        # 3. Dynamic Search Depth
        # If endgame (few ships left), search deeper
        ships_left = 5 - len(sunk_ships)
        search_depth = 1
        if ships_left <= 2:
            search_depth = 2 # Look 2 moves ahead
            top_k = 4 # Narrower beam for speed

        # 4. Simulation Loop
        flat_indices = np.argsort(current_probs.ravel())[-top_k:]
        candidates = [np.unravel_index(i, current_probs.shape) for i in flat_indices]
        
        best_score = -1
        best_move = candidates[-1] 
        
        for r, c in candidates:
            score = self._simulate_move(board_state, sunk_ships, r, c, current_probs, depth=search_depth)
            
            if score > best_score:
                best_score = score
                best_move = (r, c)
                
        return best_move

    def _simulate_move(self, board_state, sunk_ships, r, c, current_probs, depth=1):
        """Recursive simulation to calculate Information Gain."""
        p_hit = current_probs[r, c] / np.sum(current_probs)
        
        # Base Case: Depth 0 or minimal probability
        if depth == 0 or p_hit < 0.01:
            return 0 

        # SIMULATE HIT
        state_hit = board_state.copy(); state_hit[r, c] = 1 
        heatmap_hit = self.math_agent.generate_heatmap(state_hit, sunk_ships)
        certainty_hit = np.max(heatmap_hit) if np.sum(heatmap_hit) > 0 else 0
        
        # Recursive Step: If I hit, what's my NEXT best move's value?
        # (Simplified: Just add the certainty. Full recursion is too slow for Python)
        
        # SIMULATE MISS
        state_miss = board_state.copy(); state_miss[r, c] = 2 
        heatmap_miss = self.math_agent.generate_heatmap(state_miss, sunk_ships)
        certainty_miss = np.max(heatmap_miss) if np.sum(heatmap_miss) > 0 else 0
        
        # Weighted Score (Entropy Reduction)
        score = (p_hit * certainty_hit) + ((1 - p_hit) * certainty_miss)
        
        return score