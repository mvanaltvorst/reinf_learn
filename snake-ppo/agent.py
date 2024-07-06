from network import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from ppomemory import PPOMemory

device = torch.device("cpu")


class PPOAgent:
    def __init__(
        self,
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        entropy_coef=0.01,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.lr = lr

        self.actor = Actor()
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic = Critic()
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.loss_fn = nn.MSELoss()

        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, prob, val, reward, done):
        # Prob contains log probs.
        self.memory.store_memory(state, action, prob, val, reward, done)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state = state.reshape(1, *state.shape)
        dist = self.actor(state)
        # We need the value to store in the memory
        val = self.critic(state).detach().numpy().item()

        action = dist.sample()
        prob = dist.log_prob(action).item()

        action = action.detach().numpy().item()

        return action, prob, val

    def learn(self):
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                val_arr,
                reward_arr,
                done_arr,
                batches,
            ) = self.memory.generate_batches()

            values = val_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # GAE
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - done_arr[k])
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t

            advantage = torch.tensor(advantage, dtype = torch.float32)

            # Normalize the advantage which helps with training stability
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            values = torch.tensor(values, dtype = torch.float32)

            returns = advantage + values

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])

                dist = self.actor(states)
                critic_value = self.critic(states)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                entropy = dist.entropy().mean()
                actor_loss -= self.entropy_coef * entropy

                # returns = advantage[batch] + values[batch]
                # critic_loss = ((returns - critic_value)**2).mean()
                critic_loss = F.mse_loss(critic_value, returns[batch].reshape(-1, 1))

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    def save(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "gamma": self.gamma,
            "alpha": self.alpha,
            "gae_lambda": self.gae_lambda,
            "policy_clip": self.policy_clip,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "memory": self.memory,
        }
        torch.save(state, path)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path: Path | str):
        state = torch.load(path)
        agent = cls(
            gamma=state["gamma"],
            alpha=state["alpha"],
            gae_lambda=state["gae_lambda"],
            policy_clip=state["policy_clip"],
            batch_size=state["batch_size"],
            n_epochs=state["n_epochs"],
            lr=state["lr"],
        )
        agent.actor.load_state_dict(state["actor_state_dict"])
        agent.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        agent.critic.load_state_dict(state["critic_state_dict"])
        agent.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        agent.memory = state["memory"]
        print(f"Agent loaded from {path}")
        return agent