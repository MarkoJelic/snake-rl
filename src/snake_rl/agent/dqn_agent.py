import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.models.dqn_model import DQN
from snake_rl.agent.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        target_update_freq: int = 2000,
        device: str | None = None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)

        # IMPORTANT: initialize target with policy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target network is not trained

        # Optimizer + Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    # --------------------------------------------------
    # Action Selection (Epsilon-Greedy)
    # --------------------------------------------------

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, state_size)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # (1, action_size)

        return int(torch.argmax(q_values, dim=1).item())

    # --------------------------------------------------
    # Store Transition
    # --------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # --------------------------------------------------
    # Training Step
    # --------------------------------------------------

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # ----------------------------
        # Current Q-values
        # ----------------------------
        q_values = self.policy_net(states)  # (batch, action_size)

        # Select Q-values of executed actions
        current_q = q_values.gather(1, actions).squeeze(1)  # (batch,)

        # ----------------------------
        # Target Q-values
        # ----------------------------
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]

            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # ----------------------------
        # Loss
        # ----------------------------
        loss = self.criterion(current_q, target_q)

        # ----------------------------
        # Backprop
        # ----------------------------
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (best practice)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # ----------------------------
        # Target Network Update
        # ----------------------------
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # --------------------------------------------------
    # Epsilon Decay
    # --------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
