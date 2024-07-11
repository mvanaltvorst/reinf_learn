import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_size: int = 11, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, output_size)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        shared_output = self.shared(x)
        
        value = self.critic(shared_output)
        policy_logits = self.actor(shared_output)
        
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        return value, policy_log_probs