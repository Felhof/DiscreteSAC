import gym
import numpy as np
import matplotlib.pyplot as plt

from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 10
RUNS = 15
EPISODES_PER_RUN = 500

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = SACAgent(env)
    agent_results = []
    for _ in range(RUNS):
        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state = env.reset()
            done = False
            i = 0
            while not done and i < 200:
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                next_state, reward, done, info = env.step(action)
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                else:
                    episode_reward += reward
                state = next_state
            if evaluation_episode:
                run_results.append(episode_reward)

    env.close()

    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(EPISODES_PER_RUN)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(EPISODES_PER_RUN)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))

    ax = plt.gca()
    ax.plot(x_vals, results_mean, label='Discrete SAC', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
