from network import ActorCritic
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from rich.pretty import pprint
from pathlib import Path

class A2CAgent:
    def __init__(self, device, gamma=0.95, lr=1e-3, maxlen=3000):
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.maxlen = maxlen
        self.model = ActorCritic().to(device)
        self.model = torch.compile(self.model)  # Compile the model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, policy_log_probs = self.model(state_tensor)
            action = torch.exp(policy_log_probs).multinomial(1)
        return action.item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.tensor([x[0] for x in minibatch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[1] for x in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([x[3] for x in minibatch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32).to(self.device)

        values, policy_log_probs = self.model(states)
        next_values, _ = self.model(next_states)

        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = returns - values.squeeze()

        action_log_probs = policy_log_probs.gather(-1, actions)
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = -torch.exp(policy_log_probs) * policy_log_probs
        entropy = entropy.sum(dim=-1).mean()

        loss = actor_loss + 0.5 * critic_loss - 1e-3 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "gamma": self.gamma,
            "lr": self.lr,
            "maxlen": self.maxlen,
            "memory": list(self.memory),
        }, path)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path: str):
        state = torch.load(path)
        agent = cls(gamma=state["gamma"], lr=state["lr"], maxlen=state["maxlen"])
        agent.model.load_state_dict(state["model_state_dict"])
        agent.optimizer.load_state_dict(state["optimizer_state_dict"])
        agent.memory = deque(state["memory"], maxlen=agent.memory.maxlen)
        print(f"Agent loaded from {path}")
        return agent