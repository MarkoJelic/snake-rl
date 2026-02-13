from snake_rl.env.snake_env import SnakeEnv
import random

env = SnakeEnv()
state = env.reset()

done = False

while not done:
    action = random.randint(0, 2)
    state, reward, done = env.step(action)
    print("Reward:", reward)

print("Episode finished.")

# poetry run python src/scripts/test_env.py