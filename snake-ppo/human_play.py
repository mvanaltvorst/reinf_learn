from snake import SnakeGame

env = SnakeGame(10, 10, max_steps=3000)

from pynput import keyboard
import time

state = env.reset()
acc_reward = 0

done = False
while not done:
    with keyboard.Events() as events:
        event = events.get(1e6)
        if event.key == keyboard.KeyCode.from_char('w'):
            action = 0
        elif event.key == keyboard.KeyCode.from_char('d'):
            action = 1
        elif event.key == keyboard.KeyCode.from_char('s'):
            action = 2
        elif event.key == keyboard.KeyCode.from_char('a'):
            action = 3
        else:
            continue
        time.sleep(0.1)

    _, reward, done = env.step(action)
    acc_reward += reward
    print(acc_reward)
    env.render()

    if done:
        break




