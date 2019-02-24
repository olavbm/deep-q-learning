import random
import numpy as np


class Policy:
    def pick_action(action_values):
        raise NotImplementedError

    def decay_temperature():
        return NotImplementedError


class E_greedy(Policy):
    def __init__(self, config):
        self.e = config.e
        self.e_decay = config.e_decay
        self.e_min = config.e_min
        self.action_size = config.action_size

    def pick_action(self, action_values):
        if np.random.rand() <= self.e:
            return random.randrange(self.action_size)
        return np.argmax(action_values[0])

    def decay_temperature(self):
        if self.e > self.e_min:
            self.e *= self.e_decay
