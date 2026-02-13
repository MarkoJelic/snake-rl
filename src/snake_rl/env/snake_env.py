import random
import numpy as np
from typing import Tuple, List


class SnakeEnv:
    def __init__(self, grid_size: int = 10, seed: int | None = None):
        self.grid_size = grid_size
        self.rng = random.Random(seed)

        self.snake: List[Tuple[int, int]] = []
        self.direction: Tuple[int, int] = (1, 0)
        self.food: Tuple[int, int] | None = None

        self.score = 0
        self.done = False
        self.steps = 0
        self.max_steps = 0

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        raise NotImplementedError
