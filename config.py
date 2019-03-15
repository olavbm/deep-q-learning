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


train_config = Config(episodes=200, time_limit=500)
nn_config = Config(loss="mse", optimizer=Adam(lr=0.001))
memory_config = Config(batch_size=128, size=10000)
policy_config = Config(e=0.4, e_decay=0.99, e_min=0.000001)
update_config = Config(gamma=0.999)
