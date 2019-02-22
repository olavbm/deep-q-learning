# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from config import train_config
from config import nn_config
from config import memory_config
from config import policy_config
from models import Cartpole_nn
from config import Config
from memory import Ring_Buffer
from policy import E_greedy


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state_config = Config(state_size=state_size, action_size=action_size)
        self.gamma = 0.95  # discount rate
        self.model = self.build_model()
        self.memory = self.build_memory()
        self.policy = self.build_policy()

    def build_model(self):
        model = Cartpole_nn(nn_config + self.state_config)
        return model

    def build_memory(self):
        memory = Ring_Buffer(memory_config)
        return memory

    def build_policy(self):
        policy = E_greedy(policy_config + self.state_config)
        return policy

    def add_to_replay_memory(self, sarsd_tuple):
        self.memory.append(sarsd_tuple)

    def act(self, state):
        action_values = self.model.predict(state)
        action = self.policy.pick_action(action_values)
        return action

    def train_from_replay_memory(self):
        minibatch = self.memory.sample()
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.train(states, targets_f)
        # Keeping track of loss
        loss = history.history["loss"][0]
        self.policy.decay_temperature()
        return loss


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    import trainer

    trainer.train(env, agent, train_config)
