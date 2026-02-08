import torch
import numpy as np
from src.agents.probability import ProbabilityAgent
from src.model.net import DeepShipNet

class AlphaZeroAgent:
    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DeepShipNet().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval() # Set to evaluation mode
        self.math_agent = ProbabilityAgent() # We use this to generate the 3rd input channel

    def preprocess_state(self, board_state, sunk_ships):
        """
        Converts the board into a (1, 3, 10, 10) tensor.
        Channel 0: Hits (Binary)
        Channel 1: Misses (Binary)
        Channel 2: Probability Heatmap (Float)
        """
        hits = (board_state == 1).astype(np.float32)
        misses = (board_state == 2).astype(np.float32)
        
        # Generate the heatmap using the Math Agent
        heatmap = self.math_agent.generate_heatmap(board_state, sunk_ships)
        # Normalize heatmap to 0-1 range to help the Neural Network
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        # Stack into 3 channels
        state_stack = np.stack([hits, misses, heatmap])
        
        # Add batch dimension: (3, 10, 10) -> (1, 3, 10, 10)
        return torch.tensor(state_stack).unsqueeze(0).to(self.device)

    def get_action(self, board_state, sunk_ships, epsilon=0.0):
        """
        Returns the best move (row, col).
        epsilon: Probability of choosing a random move (for exploration during training)
        """
        valid_moves = (board_state == 0).astype(np.float32)
        
        # Exploration (Random move)
        if np.random.random() < epsilon:
            candidates = np.argwhere(valid_moves)
            return tuple(candidates[np.random.choice(len(candidates))])

        # Exploitation (Neural Network move)
        state_tensor = self.preprocess_state(board_state, sunk_ships)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            
        # Convert logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().squeeze()
        
        # Mask invalid moves (set prob to 0)
        valid_moves_flat = valid_moves.flatten()
        policy_probs = policy_probs * valid_moves_flat
        
        # Re-normalize
        if np.sum(policy_probs) > 0:
            policy_probs /= np.sum(policy_probs)
        else:
            # Fallback if model is confused
            return self.math_agent.get_action(board_state, sunk_ships)
            
        # Pick best move
        best_move_flat = np.argmax(policy_probs)
        best_move = np.unravel_index(best_move_flat, (10, 10))
        
        return best_move