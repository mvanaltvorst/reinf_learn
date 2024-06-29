from network import ActorCritic
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cpu")

class A2CAgent:
    def __init__(self, lr=1e-3, maxlen=2000):
        self.model = ActorCritic(2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=maxlen)

    def remember(self, action, reward):
        self.memory.append((action, reward))

    def act(self):
        _, policy_dist = self.model()
        dist = torch.distributions.Categorical(probs=policy_dist)
        action = dist.sample().detach().numpy().item()
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        actions = torch.tensor([x[0] for x in minibatch]).unsqueeze(1).to(device)
        rewards = torch.tensor([x[1] for x in minibatch]).to(
            device, dtype=torch.float32
        )

        values, policy_dists = self.model()

        # Expand to batch size
        values = values.expand(batch_size, -1)
        policy_dists = policy_dists.expand(batch_size, -1)

        advantages = (
            rewards - values.squeeze()
        )  # usually returns - values, where returns follow from the bellman eq.

        log_probs = torch.log(policy_dists.gather(1, actions))
        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = (advantages**2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
