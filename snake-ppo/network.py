import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size: int = 10, output_size: int = 4):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.actor = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        policy_dist = self.actor(x)
        policy_dist = torch.distributions.Categorical(policy_dist)
        return policy_dist


class Critic(nn.Module):
    def __init__(self, input_size: int = 10):
        super(Critic, self).__init__()
        self.input_size = input_size

        self.critic = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        value = self.critic(x)
        return value
