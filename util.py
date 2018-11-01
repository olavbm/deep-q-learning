import gym
import numpy as np

def render(agent):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    done = False
    batch_size = 32

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            return
    return

