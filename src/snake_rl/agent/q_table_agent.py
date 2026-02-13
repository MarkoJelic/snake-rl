import numpy as np
from typing import Tuple


class QTableAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table initialized to zeros
        self.q_table = {}

    def _state_to_key(self, state: np.ndarray) -> Tuple:
        return tuple(state.tolist())

    def get_action(self, state: np.ndarray) -> int:
        state_key = self._state_to_key(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        q_current = self.q_table[state_key][action]

        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state_key])

        self.q_table[state_key][action] += self.alpha * (q_target - q_current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
