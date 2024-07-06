import torch
import torch.nn as nn
import torch.nn.functional as F

# Old: hardcoded features
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


# New: CNN architecture
# class SimpleNet(nn.Module):
#     def __init__(self, input_channels: int = 3, hidden_size: int = 64):
#         super(SimpleNet, self).__init__()
#         self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
#         self.fc = nn.Linear(16 * 10 * 10, hidden_size)

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)  # Change from (B, 10, 10, 3) to (B, 3, 10, 10)
#         x = F.relu(self.conv(x))
#         x = x.reshape(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc(x))
#         return x

# class Actor(nn.Module):
#     def __init__(self, input_channels: int = 3, hidden_size: int = 64, output_size: int = 4):
#         super(Actor, self).__init__()
#         self.base = SimpleNet(input_channels, hidden_size)
#         self.actor_head = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.base(x)
#         policy_logits = self.actor_head(x)
#         return torch.distributions.Categorical(logits=policy_logits)

# class Critic(nn.Module):
#     def __init__(self, input_channels: int = 3, hidden_size: int = 64):
#         super(Critic, self).__init__()
#         self.base = SimpleNet(input_channels, hidden_size)
#         self.critic_head = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = self.base(x)
#         value = self.critic_head(x)
#         return value