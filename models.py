import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Model:
    def build(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def train(self, x, y, config):
        raise NotImplementedError


class Cartpole_nn(Model):
    def __init__(self, config):
        self.model = self.build(config)

    def build(self, config):
        model = Sequential()
        model.add(Dense(24, input_dim=config.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(config.action_size, activation="linear"))
        model.compile(loss=config.loss, optimizer=config.optimizer)
        return model

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, y, **kwargs):
        return self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)
