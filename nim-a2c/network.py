import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, max_state, output_size):
        super().__init__()
        self.max_state = max_state

        self.actor = nn.Sequential(
            nn.Linear(self.max_state + 1, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.max_state + 1, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        one_hotted = F.one_hot(x, num_classes=self.max_state + 1).float()
        value = self.critic(one_hotted)
        policy_dist = self.actor(one_hotted)
        return value, policy_dist
