import numpy as np
from queue import Queue
import torch


class SnakeGame:
    def __init__(self, width: int, height: int, max_steps: int = 1000):
        # Action space: 0 (north), 1 (east), 2 (south), 3 (west)
        self.action_space = [0, 1, 2, 3]
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        start_x, start_y = np.random.randint(self.width), np.random.randint(self.height)
        self.tail = Queue()
        self.tail.put((start_x, start_y))
        self.current_direction = np.random.choice([0, 1, 2, 3])
        self.steps = 0

        # spawn an apple on a location that does not intersect with the current head
        while True:
            self.apple_x, self.apple_y = (
                np.random.randint(self.width),
                np.random.randint(self.height),
            )
            if self.apple_x != start_x or self.apple_y != start_y:
                break

        return self._get_state()

    def step(self, action):
        assert action in self.action_space
        self.current_direction = action
        cur_x, cur_y = self.tail.queue[-1]
        if action == 0:
            cur_y -= 1
        elif action == 1:
            cur_x += 1
        elif action == 2:
            cur_y += 1
        else:
            cur_x -= 1

        self.steps += 1

        # Case: new head is outside of map
        if cur_x >= self.width or cur_x < 0 or cur_y >= self.height or cur_y < 0:
            return self._get_state(), -3, True

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
                if (self.apple_x, self.apple_y) not in self.tail.queue and cur_x != self.apple_x and cur_y != self.apple_y:
                    break
        else:
            # Get rid of far tail section
            self.tail.get()

        # Case: new head intersects tail
        if (cur_x, cur_y) in self.tail.queue:
            return self._get_state(), -3, True

        # Add new head
        self.tail.put((cur_x, cur_y))

        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            return self._get_state(), reward, True

        return self._get_state(), reward, False

    def _get_state(self):
        # head_x, head_y = self.tail.queue[-1]

        # # Current features
        # apple_delta_x = self.apple_x - head_x
        # apple_delta_y = self.apple_y - head_y

        # # New features
        # # Danger straight ahead
        # danger_straight = self._is_collision(head_x, head_y, self.current_direction)

        # # Danger to the right
        # danger_right = self._is_collision(head_x, head_y, (self.current_direction + 1) % 4)

        # # Danger to the left
        # danger_left = self._is_collision(head_x, head_y, (self.current_direction - 1) % 4)

        # # Current direction
        # dir_left = self.current_direction == 3
        # dir_right = self.current_direction == 1
        # dir_up = self.current_direction == 0
        # dir_down = self.current_direction == 2

        # # Snake length
        # snake_length = len(self.tail.queue)

        # return [
        #     apple_delta_x, apple_delta_y,
        #     danger_straight, danger_right, danger_left,
        #     dir_left, dir_right, dir_up, dir_down,
        #     snake_length
        # ]

        head_x, head_y = self.tail.queue[-1]
        neck_x, neck_y = self.tail.queue[-2]

        dx = head_x - neck_x
        dy = head_y - neck_y

        rot_matrix = np.array([[0, 1], [-1, 0]])
        left = rot_matrix @ np.array([dx, dy])
        dx_left, dy_left = left.tolist()
        right = rot_matrix.T @ np.array([dx, dy])
        dx_right, dy_right = right.tolist()

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
            head_x, head_y, dx_left, dy_left
        )
        dist_straight, is_tail_straight, is_apple_straight = find_dist(
            head_x, head_y, dx, dy
        )
        dist_right, is_tail_right, is_apple_right = find_dist(
            head_x, head_y, dx_right, dy_right
        )

        return torch.tensor(
            [
                dist_left,
                is_tail_left,
                is_apple_left,
                dist_straight,
                is_tail_straight,
                is_apple_straight,
                dist_right,
                is_tail_right,
                is_apple_right,
            ],
            dtype=torch.float32,
        )

        # Raw board state
        # state = torch.zeros((self.height, self.width, 3), dtype=torch.float32)

        # # Fill in the tail
        # for x, y in list(self.tail.queue)[:-1]:  # Exclude the head
        #     state[y, x, 0] = 1

        # # Fill in the head
        # head_x, head_y = self.tail.queue[-1]
        # state[head_y, head_x, 1] = 1

        # # Fill in the apple
        # state[self.apple_y, self.apple_x, 2] = 1

        # return state

    def _is_collision(self, x, y, direction):
        if direction == 0:  # Up
            y -= 1
        elif direction == 1:  # Right
            x += 1
        elif direction == 2:  # Down
            y += 1
        elif direction == 3:  # Left
            x -= 1

        # Check if out of bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Check if collision with snake body
        if (x, y) in self.tail.queue:
            return True

        return False

    def get_action_mask(self):
        mask = [1, 1, 1, 1]  # All actions initially allowed
        head_x, head_y = self.tail.queue[-1]

        # Prevent 180-degree turns
        if self.current_direction == 0:  # Up
            mask[2] = 0  # Can't go down
        elif self.current_direction == 1:  # Right
            mask[3] = 0  # Can't go left
        elif self.current_direction == 2:  # Down
            mask[0] = 0  # Can't go up
        elif self.current_direction == 3:  # Left
            mask[1] = 0  # Can't go right

        return torch.tensor(mask, dtype=torch.float32)

    def get_score(self):
        return len(self.tail.queue) - 1

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
