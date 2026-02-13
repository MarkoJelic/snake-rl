from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agent.q_table_agent import QTableAgent

import numpy as np
from datetime import datetime
from pathlib import Path

env = SnakeEnv()
agent = QTableAgent(state_size=11, action_size=3)

episodes = 5000

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Create timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = results_dir / f"train_{timestamp}.txt"

log_f = open(log_file, "w")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if episode % 100 == 0:
        log_line = f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
        print(log_line)
        log_f.write(log_line + "\n")

log_f.close()

# poetry run python src/scripts/train_q_table.py