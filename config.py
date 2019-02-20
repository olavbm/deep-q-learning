from keras.optimizers import Adam


class Config(dict):
    def __getattr__(self, name):
        if name[0] == "_":
            raise AttributeError(name)
        return self[name]

    """
    def __repr__(self):
        return f"Config({', '.join(k + '=' + repr(v) for k,v in self.items())})"
    """

    def __add__(self, other):
        return Config(**{**self, **other})


train_config = Config(episodes=100, time_limit=500)
nn_config = Config(loss="mse", optimizer=Adam(lr=0.001))
