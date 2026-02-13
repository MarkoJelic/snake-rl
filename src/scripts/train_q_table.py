from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agent.q_table_agent import QTableAgent

import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

env = SnakeEnv()
agent = QTableAgent(state_size=11, action_size=3)

episodes = 5000

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Create timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = results_dir / f"train_{timestamp}.txt"

episode_rewards = []
moving_averages = []
epsilon_values = []

eval_rewards = []
eval_checkpoints = []

window_size = 100

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

    episode_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)

    # Compute moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.mean(episode_rewards[-window_size:])
    else:
        moving_avg = np.mean(episode_rewards)

    moving_averages.append(moving_avg)

    if episode % 500 == 0 and episode > 0:
        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0  # pure greedy

        eval_total = 0
        eval_runs = 20

        for _ in range(eval_runs):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
                episode_reward += reward

            eval_total += episode_reward

        avg_eval_reward = eval_total / eval_runs

        eval_rewards.append(avg_eval_reward)
        eval_checkpoints.append(episode)

        agent.epsilon = saved_epsilon  # restore

        print(f"Evaluation at {episode}: {avg_eval_reward:.2f}")


    if episode % 100 == 0:
        log_line = f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
        print(log_line)
        log_f.write(log_line + "\n")

log_f.close()

plt.figure(figsize=(10, 6))

plt.plot(episode_rewards, label="Raw Reward", alpha=0.3)
plt.plot(moving_averages, label="Moving Average (100)")

plt.scatter(eval_checkpoints, eval_rewards, label="Evaluation", color="red")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("Training Performance")
plot_path = results_dir / f"training_plot_{timestamp}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(epsilon_values)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
epsilon_plot_path = results_dir / f"epsilon_plot_{timestamp}.png"
plt.savefig(epsilon_plot_path, dpi=150, bbox_inches="tight")
plt.close()

# poetry run python src/scripts/train_q_table.py