from typing import Dict, List
import numpy as np


def train_dqn(
    env,
    agent,
    episodes: int = 5000,
    eval_interval: int = 500,
    eval_episodes: int = 20,
    moving_avg_window: int = 100,
) -> Dict[str, List[float]]:
    episode_rewards = []
    moving_averages = []
    epsilon_values = []
    losses = []

    eval_rewards = []
    eval_checkpoints = []

    total_env_steps = 0
    print("Using device:", agent.device)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            total_env_steps += 1

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        # Moving average
        if len(episode_rewards) >= moving_avg_window:
            moving_avg = np.mean(episode_rewards[-moving_avg_window:])
        else:
            moving_avg = np.mean(episode_rewards)

        moving_averages.append(moving_avg)

        # ----------------------------
        # Evaluation (greedy policy)
        # ----------------------------
        if episode % eval_interval == 0 and episode > 0:
            saved_epsilon = agent.epsilon
            agent.epsilon = 0.0

            eval_total = 0.0

            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    episode_reward += reward

                eval_total += episode_reward

            avg_eval_reward = eval_total / eval_episodes

            eval_rewards.append(avg_eval_reward)
            eval_checkpoints.append(episode)

            agent.epsilon = saved_epsilon

        # Optional lightweight progress print
        if episode % 100 == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.2f} | "
                f"Moving Avg: {moving_avg:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return {
        "episode_rewards": episode_rewards,
        "moving_averages": moving_averages,
        "epsilon_values": epsilon_values,
        "losses": losses,
        "eval_rewards": eval_rewards,
        "eval_checkpoints": eval_checkpoints,
    }
