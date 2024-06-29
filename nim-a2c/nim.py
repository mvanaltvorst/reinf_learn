import numpy as np

class Nim21Game:
    def __init__(self):
        # Action space: remove 1, 2, or 3 objects
        self.action_space = [1, 2, 3]
        
        # Observation space: number of objects remaining (0-21)
        self.observation_space = list(range(1, 22))
        
        self.reset()

    def reset(self):
        self.remaining = 21
        return self.remaining

    def step(self, action):
        assert action in self.action_space

        # Convert action to number of objects removed (action 0 -> remove 1, action 1 -> remove 2, action 2 -> remove 3)
        self.remaining -= action
        
        # Check if the game is over
        done = self.remaining <= 0
        
        # Reward: 1 for winning, -1 for losing, 0 otherwise
        if done:
            reward = 1 if self.remaining == 0 else -1
            self.remaining = 0
        else:
            reward = 0
        
        return self.remaining, reward, done

    def render(self):
        print(f"Remaining objects: {self.remaining}")
