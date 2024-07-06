import numpy as np
from queue import Queue
import random


class SnakeGame:
    def __init__(self, width: int, height: int, max_steps: int = 2000):
        # Action space: 0 (left), 1 (straight), 2 (right)
        self.action_space = [0, 1, 2]
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # Place the head, ensuring there's space for a second segment
        head_x = np.random.randint(1, self.width - 1)
        head_y = np.random.randint(1, self.height - 1)
        self.tail = Queue()
        self.tail.put((head_x, head_y))

        # Find valid positions for the second segment
        valid_positions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = random.choice(valid_positions)
        second_x, second_y = head_x + dx, head_y + dy
        self.tail.put((second_x, second_y))

        self.direction = (dx, dy)  # Current direction
        self.steps = 0

        # Spawn an apple on a location that does not intersect with the current snake
        while True:
            self.apple_x, self.apple_y = (
                np.random.randint(self.width),
                np.random.randint(self.height),
            )
            if (self.apple_x, self.apple_y) not in self.tail.queue:
                break

        return self._get_state()

    def step(self, action):
        assert action in self.action_space

        # Convert relative action to new direction
        if action == 0:  # Turn left
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # Turn right
            self.direction = (self.direction[1], -self.direction[0])
        # action == 1 means go straight, so we don't change direction

        cur_x, cur_y = self.tail.queue[-1]
        cur_x += self.direction[0]
        cur_y += self.direction[1]

        self.steps += 1

        # Case: new head is outside of map
        if cur_x >= self.width or cur_x < 0 or cur_y >= self.height or cur_y < 0:
            return self._get_state(), -1, True

        reward = 0
        # Case: head intersects apple
        if cur_x == self.apple_x and cur_y == self.apple_y:
            reward += 1
            # move apple to somewhere that doesn't intersect with the snake
            while True:
                self.apple_x, self.apple_y = (
                    np.random.randint(self.width),
                    np.random.randint(self.height),
                )
                if (self.apple_x, self.apple_y) not in self.tail.queue:
                    break
        else:
            # Get rid of far tail section
            self.tail.get()

        # Case: new head intersects tail
        if (cur_x, cur_y) in self.tail.queue:
            return self._get_state(), -1, True

        # Add new head
        self.tail.put((cur_x, cur_y))

        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    def _get_state(self):
        head_x, head_y = self.tail.queue[-1]
        apple_x, apple_y = self.apple_x, self.apple_y

        # Define the relative directions
        straight = self.direction
        left = (-self.direction[1], self.direction[0])
        right = (self.direction[1], -self.direction[0])

        diff_x, diff_y = apple_x - head_x, apple_y - head_y

        # check the relative distance w.r.t. the direction
        straight_apple_cos = straight[0] * diff_x + straight[1] * diff_y
        left_apple_cos = left[0] * diff_x + left[1] * diff_y

        # Cosine similarity needs to be normalised.
        # Add 1 to the sqrt denominator to avoid division by zero
        straight_apple_cos /= np.sqrt(
            (diff_x**2 + diff_y**2) * (straight[0] ** 2 + straight[1] ** 2) + 1
        )
        left_apple_cos /= np.sqrt(
            (diff_x**2 + diff_y**2) * (left[0] ** 2 + left[1] ** 2) + 1
        )

        def find_dist(
            cur_x, cur_y, dx, dy
        ) -> tuple[int, bool, bool]:  # dist, is_tail, is_apple
            dist = 0
            is_tail = False
            is_apple = False
            while True:
                cur_x += dx
                cur_y += dy
                dist += 1
                if (
                    cur_x < 0
                    or cur_x >= self.width
                    or cur_y < 0
                    or cur_y >= self.height
                ):
                    break
                if (cur_x, cur_y) in self.tail.queue:
                    is_tail = True
                    break
                if cur_x == self.apple_x and cur_y == self.apple_y:
                    is_apple = True
                    break

            return dist, is_tail, is_apple

        dist_left, is_tail_left, is_apple_left = find_dist(
            head_x, head_y, left[0], left[1]
        )
        dist_straight, is_tail_straight, is_apple_straight = find_dist(
            head_x, head_y, straight[0], straight[1]
        )
        dist_right, is_tail_right, is_apple_right = find_dist(
            head_x, head_y, right[0], right[1]
        )

        return [
            1 / dist_left,  # Easy normalisation
            is_tail_left,
            is_apple_left,
            1 / dist_straight,  # Easy normalisation
            is_tail_straight,
            is_apple_straight,
            1 / dist_right,  # Easy normalisation
            is_tail_right,
            is_apple_right,
            # Direction to apple
            straight_apple_cos,
            left_apple_cos,
        ]

    def get_score(self):
        return len(self.tail.queue) - 2

    def render(self):
        mapping = {
            0: ".",
            1: "#",  # tail
            2: "H",  # head
            3: "0",  # apple
            4: "*",  # border
        }
        # start with an empty matrix
        matrix = np.zeros(shape=(self.height, self.width), dtype=np.uint8)
        # add the tail
        for x, y in self.tail.queue:
            matrix[y, x] = 1
        # add the head
        cur_x, cur_y = self.tail.queue[-1]
        matrix[cur_y, cur_x] = 2
        # add the apple
        matrix[self.apple_y, self.apple_x] = 3
        # add a border
        padded = np.pad(matrix, 1, "constant", constant_values=4)
        for row in padded:
            for val in row:
                print(mapping[val], end="")
            print("")
        print("")
