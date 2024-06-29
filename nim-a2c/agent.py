from network import ActorCritic
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from rich.pretty import pprint

device = torch.device("cpu")


class A2CAgent:
    def __init__(self, gamma=0.95, lr=1e-3, maxlen=2000):
        self.gamma = gamma

        self.model = ActorCritic(21, 3)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        tensor_state = torch.tensor([state]).to(device)
        _, policy_dist = self.model(tensor_state)
        # action = policy_dist.sample() + 1
        # action = np.random.choice(list(range(3)), p = policy_dist.detach().numpy().squeeze()) + 1
        dist = torch.distributions.Categorical(probs=policy_dist)
        action = dist.sample().detach().numpy()[0] + 1
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.tensor([x[0] for x in minibatch]).to(device)
        actions = torch.tensor([x[1] for x in minibatch]).unsqueeze(1).to(device)
        rewards = torch.tensor([x[2] for x in minibatch]).to(
            device, dtype=torch.float32
        )
        next_states = torch.tensor([x[3] for x in minibatch]).to(device)
        dones = torch.tensor([x[4] for x in minibatch]).to(device, dtype=torch.float32)

        values, policy_dists = self.model(states)
        next_values, _ = self.model(next_states)

        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = returns - values.squeeze()

        log_probs = torch.log(policy_dists.gather(1, actions - 1))
        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = (advantages**2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
