import numpy as np


class Rule:
    def train(model, minibatch):
        raise NotImplementedError

    def __call__():
        raise NotImplementedError


class Q_learning:
    def __init__(self, config):
        self.gamma = config.gamma

    def __call__(self, model, minibatch):
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Bellman equation
                target = reward + self.gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = model.train(states, targets_f)
        return history
