import torch
import torch.nn as nn
import torch.nn.functional as F


# Deep Q Network
class DQN(nn.Module):
    def __init__(self, max_state, output_size):
        super().__init__()
        self.max_state = max_state

        self.seq = nn.Sequential(
            nn.Linear(self.max_state + 1, 32), nn.ReLU(), nn.Linear(32, output_size)
        )

    def forward(self, x):
        one_hotted = F.one_hot(x, num_classes=self.max_state + 1).float()
        return self.seq(one_hotted)
