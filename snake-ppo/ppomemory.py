import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        # We need to store the probs as well because PPO uses an "off-policy learning approach"
        # need the old probs to compute the importance sampling ratio, allowing PPO to make conservative updates.
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        # Iterate over history in batches
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # Random non-overlapping batches spanning the entire history
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
