import numpy as np
import yaml
import pickle
import os
import random
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
        """
        Loads a pre-calculated 'Survivor' layout if available.
        Otherwise falls back to random.
        """
        layout_file = "data/best_layouts.pkl"
        if os.path.exists(layout_file):
            try:
                with open(layout_file, "rb") as f:
                    layouts = pickle.load(f)
                
                # Pick one at random
                chosen_layout = random.choice(layouts)
                self._apply_layout(chosen_layout)
                return
            except Exception as e:
                print(f"Error loading layouts: {e}. Using random.")
        
        self.place_ship_randomly_pure()

    def place_ship_randomly_pure(self):
        """Standard random placement (used for generating the initial pool)."""
        self.reset()
        for name, size in self.ship_templates.items():
            ship = Ship(name, size)
            placed = False
            while not placed:
                placed = self._attempt_place_ship_virtual(ship, set(self.ship_map.keys()))
                if placed:
                    self.ships.append(ship)
                    for r, c in ship.coords:
                        self.ship_map[(r, c)] = ship

    def _apply_layout(self, layout_config: dict):
        """Applies a specific ship configuration."""
        self.reset()
        for name, coords in layout_config.items():
            size = self.ship_templates[name]
            ship = Ship(name, size)
            # Convert list of lists/tuples to tuples
            tuple_coords = [tuple(c) for c in coords]
            ship.place(tuple_coords)
            
            self.ships.append(ship)
            for r, c in ship.coords:
                self.ship_map[(r, c)] = ship

    def _attempt_place_ship_virtual(self, ship: Ship, occupied: set) -> bool:
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
            if (r, c) in occupied: return False
            
        ship.place(coords)
        return True

    # --- Standard Methods ---
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
        for r, c in ship.coords: self.ship_map[(r, c)] = ship
        return True

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
        for r, c in ship.coords: self.state[r, c] = 3 

    def is_game_over(self) -> bool:
        return self.sunk_ships == len(self.ship_templates)