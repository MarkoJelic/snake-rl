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
        center = self.grid_size // 2

        # Initialize snake in the center, moving right
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]

        self.direction = (1, 0)  # moving right

        self.score = 0
        self.done = False
        self.steps = 0
        self.max_steps = 100 * len(self.snake)

        self._spawn_food()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("Cannot call step() on terminated episode. Call reset().")

        self.steps += 1

        # Update direction
        self._rotate_direction(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Check collision
        if self._is_collision(new_head):
            self.done = True
            reward = -10.0
            return self._get_state(), reward, self.done

        # Insert new head
        self.snake.insert(0, new_head)

        reward = -0.01  # step penalty

        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self._spawn_food()
            self.max_steps = 100 * len(self.snake)
        else:
            # Remove tail
            self.snake.pop()

        # Timeout condition
        if self.steps >= self.max_steps:
            self.done = True
            reward = -10.0

        return self._get_state(), reward, self.done


    def _spawn_food(self) -> None:
        while True:
            x = self.rng.randint(0, self.grid_size - 1)
            y = self.rng.randint(0, self.grid_size - 1)
            position = (x, y)

            if position not in self.snake:
                self.food = position
                return

    def _is_collision(self, position: Tuple[int, int]) -> bool:
        x, y = position

        # Wall collision
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True

        # Self collision
        if position in self.snake[1:]:
            return True

        return False

    def _get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]

        dir_x, dir_y = self.direction

        # Possible moves
        straight = (head_x + dir_x, head_y + dir_y)
        right = (head_x + dir_y, head_y - dir_x)
        left = (head_x - dir_y, head_y + dir_x)

        danger_straight = int(self._is_collision(straight))
        danger_right = int(self._is_collision(right))
        danger_left = int(self._is_collision(left))

        moving_up = int(self.direction == (0, -1))
        moving_down = int(self.direction == (0, 1))
        moving_left = int(self.direction == (-1, 0))
        moving_right = int(self.direction == (1, 0))

        food_left = int(self.food[0] < head_x)
        food_right = int(self.food[0] > head_x)
        food_up = int(self.food[1] < head_y)
        food_down = int(self.food[1] > head_y)

        state = [
            danger_straight,
            danger_right,
            danger_left,
            moving_up,
            moving_down,
            moving_left,
            moving_right,
            food_left,
            food_right,
            food_up,
            food_down,
        ]

        return np.array(state, dtype=int)

    def _rotate_direction(self, action: int) -> None:
        if action not in [0, 1, 2]:
            raise ValueError("Action must be 0 (straight), 1 (right), or 2 (left)")

        dx, dy = self.direction

        if action == 0:
            # straight
            return

        elif action == 1:
            # right turn: (dx, dy) -> (dy, -dx)
            self.direction = (dy, -dx)

        elif action == 2:
            # left turn: (dx, dy) -> (-dy, dx)
            self.direction = (-dy, dx)
