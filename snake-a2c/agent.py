from network import ActorCritic
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from rich.pretty import pprint
from pathlib import Path
import pickle

device = torch.device("cpu")


class A2CAgent:
    def __init__(self, gamma=0.95, lr=1e-3, maxlen=3000):
        self.gamma = gamma
        self.lr = lr
        self.maxlen = maxlen

        self.model = ActorCritic()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        _, policy_dist = self.model(torch.tensor(state, dtype = torch.float32))
        dist = torch.distributions.Categorical(probs=policy_dist)
        action = dist.sample().detach().numpy().item()
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.tensor([x[0] for x in minibatch], dtype = torch.float32).to(device)
        actions = torch.tensor([x[1] for x in minibatch]).unsqueeze(1).to(device)
        rewards = torch.tensor([x[2] for x in minibatch]).to(
            device, dtype=torch.float32
        )
        next_states = torch.tensor([x[3] for x in minibatch], dtype = torch.float32).to(device)
        dones = torch.tensor([x[4] for x in minibatch]).to(device, dtype=torch.float32)

        values, policy_dists = self.model(states)
        next_values, _ = self.model(next_states)

        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = returns - values.squeeze()

        log_probs = torch.log(policy_dists.gather(1, actions))
        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = (advantages**2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'lr': self.lr,
            'maxlen': self.maxlen,
            'memory': list(self.memory)
        }
        torch.save(state, path)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path: Path | str):
        state = torch.load(path)
        agent = cls(gamma=state['gamma'], lr=state['lr'], maxlen = state['maxlen'])
        agent.model.load_state_dict(state['model_state_dict'])
        agent.optimizer.load_state_dict(state['optimizer_state_dict'])
        agent.memory = deque(state['memory'], maxlen=agent.memory.maxlen)
        print(f"Agent loaded from {path}")
        return agent 