from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass
class Ship:
    name: str
    size: int
    hits: int = 0
    coords: Set[Tuple[int, int]] = None

    def __post_init__(self):
        if self.coords is None:
            self.coords = set()

    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.size

    def place(self, coords: List[Tuple[int, int]]):
        """Assigns coordinates to the ship."""
        if len(coords) != self.size:
            raise ValueError(f"{self.name} expects {self.size} coords, got {len(coords)}")
        self.coords = set(coords)

    def receive_hit(self) -> bool:
        """Registers a hit. Returns True if this hit sank the ship."""
        self.hits += 1
        return self.is_sunk