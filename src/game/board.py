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
        """Resets the board for a new game."""
        # 0 = Unknown/Empty, 1 = Hit, 2 = Miss, 3 = Sunk Marker
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        self.ships: List[Ship] = []
        self.ship_map: Dict[Tuple[int, int], Ship] = {} # Map coord to ship object
        self.shots_fired = 0
        self.hits = 0
        self.sunk_ships = 0
    
    def place_randomly(self):
        """Randomly places all ships on the board."""
        self.reset()
        for name, size in self.ship_templates.items():
            placed = False
            while not placed:
                ship = Ship(name, size)
                placed = self._attempt_place_ship(ship)
                if placed:
                    self.ships.append(ship)
                    for r, c in ship.coords:
                        self.ship_map[(r, c)] = ship

    def _attempt_place_ship(self, ship: Ship) -> bool:
        """Tries to place a single ship randomly. Returns True if successful."""
        orientation = np.random.choice(["H", "V"])
        
        if orientation == "H":
            row = np.random.randint(0, self.size)
            col = np.random.randint(0, self.size - ship.size + 1)
            coords = [(row, col + i) for i in range(ship.size)]
        else:
            row = np.random.randint(0, self.size - ship.size + 1)
            col = np.random.randint(0, self.size)
            coords = [(row + i, col) for i in range(ship.size)]

        # Check for collision
        for r, c in coords:
            if (r, c) in self.ship_map:
                return False
        
        # Place ship
        ship.place(coords)
        return True

    def fire(self, row: int, col: int) -> Tuple[bool, bool, str]:
        """
        Fires at a coordinate.
        Returns: (is_hit, is_sunk, ship_name)
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False, False, "OutOfBounds"
        
        if self.state[row, col] != 0:
            return False, False, "AlreadyFired" # Penalty for repeated shots
        
        self.shots_fired += 1
        
        if (row, col) in self.ship_map:
            # HIT
            ship = self.ship_map[(row, col)]
            is_sunk = ship.receive_hit()
            self.state[row, col] = 1 # Mark as Hit
            self.hits += 1
            
            if is_sunk:
                self.sunk_ships += 1
                self._mark_sunk(ship)
                return True, True, ship.name
            
            return True, False, ship.name
        else:
            # MISS
            self.state[row, col] = 2 # Mark as Miss
            return False, False, None

    def _mark_sunk(self, ship: Ship):
        """Mark all parts of a sunk ship visually (optional logic for AI input)"""
        for r, c in ship.coords:
            self.state[r, c] = 3 # 3 represents a confirmed sunk ship part

    def get_mask(self) -> np.ndarray:
        """Returns a binary mask of valid moves (0 = valid, 1 = invalid/already fired)"""
        return (self.state != 0).astype(np.float32)

    def is_game_over(self) -> bool:
        return self.sunk_ships == len(self.ship_templates)