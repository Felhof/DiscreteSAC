import gym

from Discrete_SAC_Agent import SACAgent


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = SACAgent(env)
    state = env.reset()
    for _ in range(5000):
        action = agent.get_next_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train_on_transition(state, action, next_state, reward, done)
        state = next_state
        if done:
            state = env.reset()

    state = env.reset()
    total_reward = 0
    for _ in range(2000):
        env.render()
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    env.close()

    print("Total Reward: ", total_reward)
