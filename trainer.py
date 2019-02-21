import numpy as np
from util import render


def train(env, agent, config):
    state_size = env.observation_space.shape[0]
    for e in range(config.episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        if (e % 20) == 0:
            render(agent)

        for time in range(config.time_limit):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.add_to_replay_memory((state, action, reward, next_state, done))
            state = next_state

            if done:
                print(
                    "episode: {:4}/{}, score: {:4}, e: {:.2}".format(
                        e, config.episodes, time, agent.epsilon
                    )
                )
                break
            agent.train_from_replay_memory()
