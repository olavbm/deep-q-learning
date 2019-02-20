from pandas.compat.numpy import np

from util import render


def train(env, agent, EPISODES):
    batch_size = 32
    state_size = env.observation_space.shape[0]
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        if (e % 20) == 0:
            render(agent)

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.add_to_replay_memory((state, action, reward, next_state, done))
            state = next_state

            if done:
                print(
                    "episode: {:4}/{}, score: {:4}, e: {:.2}".format(
                        e, EPISODES, time, agent.epsilon
                    )
                )
                break

            if len(agent.memory) > batch_size:
                agent.train_from_replay_memory(batch_size)
