import numpy as np


def init_theta(size):
    return np.random.randn(size)

def normalize(x):
    max = x.max(axis = 0).reshape(1, -1)
    min = x.min(axis = 0).reshape(1, -1)
    return np.nan_to_num((x - min) / (max - min)), min, max - min

def stardardize(x):
    mean = x.mean(axis = 0).reshape(1, -1)
    std = x.std(axis = 0).reshape(1, -1)
    return np.nan_to_num((x - mean) / std), mean, std

def add_bias(x):
    if 1 in x.shape or len(x.shape) == 1:
        return np.vstack((np.ones((1)), x.reshape(x.shape[0], -1)))
    return np.hstack((np.ones((x.shape[0], 1)), x.reshape(x.shape[0], -1)))
