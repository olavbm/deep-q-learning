import gym
import time
import numpy as np


def render(current_agent):
    while True:
        # print(current_agent)
        # print(current_agent.value)
        # print(current_agent.get())
        # agent = current_agent.get()
        agent = current_agent
        # print(agent, "agent")
        env = gym.make("Pendulum-v0")
        state_size = env.observation_space.shape[0]
        action_size = 1
        done = False
        batch_size = 32

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for t in range(500):
            env.render()
            action = agent.act(state)
            next_state, _, done, _ = env.step([action])
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                break
        time.sleep(1.0)
        print("Hello from render sleep")
