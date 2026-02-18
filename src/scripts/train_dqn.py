from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agent.dqn_agent import DQNAgent
from snake_rl.training.trainer import train_dqn


def main():
    # ----------------------------
    # Environment
    # ----------------------------
    env = SnakeEnv(state_mode="grid")

    # ----------------------------
    # Agent
    # ----------------------------
    initial_state = env.reset()
    state_shape = initial_state.shape

    agent = DQNAgent(
        state_size=state_shape,
        action_size=3,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100_000,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        target_update_freq=2000,
    )

    # ----------------------------
    # Train
    # ----------------------------
    results = train_dqn(
        env=env,
        agent=agent,
        episodes=8000,
        eval_interval=500,
        eval_episodes=20,
        moving_avg_window=100,
    )

    # ----------------------------
    # Prepare Results Directory
    # ----------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------
    # Plot Training Curve
    # ----------------------------
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 6))

    # Moving average (training)
    plt.plot(
        results["moving_averages"],
        label="Training Moving Avg",
        linewidth=2,
    )

    # Evaluation
    plt.plot(
        results["eval_checkpoints"],
        results["eval_rewards"],
        label="Evaluation (Greedy)",
        linewidth=2,
        marker="o",
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training vs Evaluation Performance")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = results_dir / f"dqn_training_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nTraining complete.")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

# poetry run python src/scripts/train_dqn.py