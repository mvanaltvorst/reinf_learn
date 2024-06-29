class MatchingPenniesGame:
    def __init__(self):
        # Action space: throw 0 or 1
        self.action_space = [0, 1]

    def step(self, action_a, action_b):
        assert action_a in self.action_space
        assert action_b in self.action_space

        if action_a == action_b: # player A wants matching pennies
            return 2, -2
        else: # player B wants non-matching pennies
            return -1, 1
