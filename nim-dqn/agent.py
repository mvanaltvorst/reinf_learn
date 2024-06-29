from network import DQN
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cpu")


class DQNAgent:
    def __init__(
        self,
        gamma=0.95,
        epsilon=1,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        lr=1e-3,
        maxlen=2000,
    ):
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = DQN(21, 3).to(device)
        self.target_net = DQN(21, 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice(list(range(1, 1 + min(state, 3))))
        else:
            with torch.no_grad():
                tensor_state = torch.tensor([state]).to(device)
                act_values = self.policy_net(tensor_state)
                return torch.argmax(act_values[0]).item() + 1

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

        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions - 1)

        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_max_q_values = torch.max(next_q_values, dim=1).values.detach()

        target_q_values = rewards + (self.gamma * next_max_q_values * (1 - dones))

        # Compute the loss
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
