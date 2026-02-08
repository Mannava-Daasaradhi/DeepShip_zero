import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

class ProbabilityAgent:
    def __init__(self, board_size=10, ships: Dict[str, int] = None, device="cuda"):
        self.board_size = board_size
        self.ships = ships if ships else {
            "Carrier": 5, "Battleship": 4, "Cruiser": 3, "Submarine": 3, "Destroyer": 2
        }
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def get_action(self, board_state: np.ndarray, sunk_ships: List[str]) -> Tuple[int, int]:
        heatmap = self.generate_heatmap(board_state, sunk_ships)
        
        # Move back to CPU for selection logic
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
            
        # Mask out taken spots
        mask = (board_state == 0)
        heatmap = heatmap * mask
        
        if np.sum(heatmap) == 0:
            valid_moves = np.argwhere(board_state == 0)
            if len(valid_moves) == 0: return (0, 0) 
            return tuple(valid_moves[np.random.choice(len(valid_moves))])
            
        best_move_flat = np.argmax(heatmap)
        return np.unravel_index(best_move_flat, heatmap.shape)

    def generate_heatmap(self, board_state: np.ndarray, sunk_ships: List[str]) -> np.ndarray:
        # Prepare Tensors on GPU
        # Board: 0=Unknown, 1=Hit, 2=Miss, 3=Sunk
        board_tensor = torch.tensor(board_state, device=self.device, dtype=torch.float32)
        
        # 1. Identify Obstacles (Misses=2, Sunk=3)
        obstacles = ((board_tensor == 2) | (board_tensor == 3)).float().unsqueeze(0).unsqueeze(0)
        
        # 2. Identify Hits (Hits=1)
        hits = (board_tensor == 1).float().unsqueeze(0).unsqueeze(0)
        
        total_heatmap = torch.zeros((self.board_size, self.board_size), device=self.device)
        
        active_ships = {name: size for name, size in self.ships.items() if name not in sunk_ships}
        
        for name, size in active_ships.items():
            # Kernels
            k_h = torch.ones((1, 1, 1, size), device=self.device)
            k_v = torch.ones((1, 1, size, 1), device=self.device)
            
            # --- Horizontal ---
            invalid_h = F.conv2d(obstacles, k_h, padding=0)
            hits_h = F.conv2d(hits, k_h, padding=0)
            weight_h = (1 + (100 * hits_h)) * (invalid_h == 0).float()
            
            # Scatter back
            total_heatmap += F.conv_transpose2d(weight_h, k_h).squeeze()

            # --- Vertical ---
            invalid_v = F.conv2d(obstacles, k_v, padding=0)
            hits_v = F.conv2d(hits, k_v, padding=0)
            weight_v = (1 + (100 * hits_v)) * (invalid_v == 0).float()
            
            total_heatmap += F.conv_transpose2d(weight_v, k_v).squeeze()

        return total_heatmap.cpu().numpy()