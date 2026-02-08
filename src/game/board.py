import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict
from src.game.ships import Ship

class Board:
    def __init__(self, config_path="config/game_settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.size = self.config["board_size"]
        self.ship_templates = self.config["ships"]
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        self.ships: List[Ship] = []
        self.ship_map: Dict[Tuple[int, int], Ship] = {} 
        self.shots_fired = 0
        self.hits = 0
        self.sunk_ships = 0
    
    def place_randomly(self):
        """Falls back to optimized placement for AI."""
        self.place_optimized()

    def place_optimized(self, iterations=500):
        """
        Generates multiple layouts and picks the one with the best 'Survival Score'.
        Survival Score = Distance between ships + Edge Penalty (Humans check edges last?)
        Actually, best metric is: Minimizing 'clumping' and maximizing 'entropy'.
        """
        best_layout_ships = []
        best_score = -1
        
        # Center of board
        center = self.size / 2.0
        
        for _ in range(iterations):
            # 1. Generate a random layout
            temp_ships = []
            temp_map = set()
            valid_layout = True
            
            for name, size in self.ship_templates.items():
                ship = Ship(name, size)
                # Try to place this ship
                placed = False
                for attempt in range(20): # Try 20 times to fit this ship
                    if self._attempt_place_ship_virtual(ship, temp_map):
                        temp_ships.append(ship)
                        placed = True
                        break
                if not placed:
                    valid_layout = False
                    break
            
            if valid_layout:
                # 2. Score this layout
                # Metric: Sum of distances between all ships (Spread them out!)
                # If ships are far apart, finding one doesn't help find others.
                score = 0
                centers = []
                for s in temp_ships:
                    # Calculate ship center
                    rs, cs = zip(*list(s.coords))
                    r_mean, c_mean = np.mean(rs), np.mean(cs)
                    centers.append((r_mean, c_mean))
                
                # Sum of distances between all pairs
                import itertools
                for c1, c2 in itertools.combinations(centers, 2):
                    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    score += dist
                
                # Bonus: Avoid touching edges (Humans look there?) 
                # Or Penalty? Actually, edges are good hiding spots against probability density.
                # Let's trust pure spacing.
                
                if score > best_score:
                    best_score = score
                    best_layout_ships = temp_ships

        # 3. Apply the best layout
        self.reset()
        for ship in best_layout_ships:
            self.ships.append(ship)
            for r, c in ship.coords:
                self.ship_map[(r, c)] = ship

    def _attempt_place_ship_virtual(self, ship: Ship, occupied: set) -> bool:
        """Helper for optimized placement (doesn't modify actual board)."""
        orientation = np.random.choice(["H", "V"])
        
        if orientation == "H":
            row = np.random.randint(0, self.size)
            col = np.random.randint(0, self.size - ship.size + 1)
            coords = [(row, col + i) for i in range(ship.size)]
        else:
            row = np.random.randint(0, self.size - ship.size + 1)
            col = np.random.randint(0, self.size)
            coords = [(row + i, col) for i in range(ship.size)]

        for r, c in coords:
            if (r, c) in occupied:
                return False
            # Optional: Don't touch other ships (Leave 1 gap)
            # This reduces valid search space for Human! 
            # If we enforce gaps, human knows "If hit, neighbor is empty".
            # So NO GAP enforcement is actually better for AI survival.

        ship.place(coords)
        occupied.update(coords)
        return True

    def place_ship_manual(self, name: str, row: int, col: int, orientation: str) -> bool:
        size = self.ship_templates[name]
        if orientation == "H": coords = [(row, col + i) for i in range(size)]
        else: coords = [(row + i, col) for i in range(size)]

        for r, c in coords:
            if not (0 <= r < self.size and 0 <= c < self.size): return False
        for r, c in coords:
            if (r, c) in self.ship_map: return False

        ship = Ship(name, size)
        ship.place(coords)
        self.ships.append(ship)
        for r, c in ship.coords:
            self.ship_map[(r, c)] = ship
        return True

    def _attempt_place_ship(self, ship: Ship) -> bool:
        # Legacy wrapper
        return self._attempt_place_ship_virtual(ship, set(self.ship_map.keys()))

    def fire(self, row: int, col: int) -> Tuple[bool, bool, str]:
        if not (0 <= row < self.size and 0 <= col < self.size): return False, False, "OutOfBounds"
        if self.state[row, col] != 0: return False, False, "AlreadyFired" 
        
        self.shots_fired += 1
        
        if (row, col) in self.ship_map:
            ship = self.ship_map[(row, col)]
            is_sunk = ship.receive_hit()
            self.state[row, col] = 1 
            self.hits += 1
            if is_sunk:
                self.sunk_ships += 1
                self._mark_sunk(ship)
                return True, True, ship.name
            return True, False, ship.name
        else:
            self.state[row, col] = 2 
            return False, False, None

    def _mark_sunk(self, ship: Ship):
        for r, c in ship.coords:
            self.state[r, c] = 3 

    def is_game_over(self) -> bool:
        return self.sunk_ships == len(self.ship_templates)