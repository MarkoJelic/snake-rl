import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_shape, action_size: int):
        super(DQN, self).__init__()

        channels, height, width = state_shape

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)

        # Shared FC
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Linear(256, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_shared(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
