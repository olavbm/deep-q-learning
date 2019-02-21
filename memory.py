from collections import deque
import random


class Memory:
    def append(self, x):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError


class Ring_Buffer(Memory):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.memory = deque(maxlen=config.size)

    def append(self, x):
        self.memory.append(x)

    def sample(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def __len__(self):
        return self.memory.maxlen
