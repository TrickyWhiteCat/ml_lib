import numpy as np


def init_theta(size):
    return np.random.randn(size)

def normalize(x):
    x = np.array(x)
    max = x.max(axis = 0).reshape(1, -1)
    min = x.min(axis = 0).reshape(1, -1)
    return np.nan_to_num((x - min) / (max - min)), min, max - min

def standardize(x):
    x = np.array(x)
    mean = x.mean(axis = 0).reshape(1, -1)
    std = x.std(axis = 0).reshape(1, -1)
    return np.nan_to_num((x - mean) / std), mean, std

def add_bias(x):
    if 1 == x.shape[1] or len(x.shape) == 1:
        return np.vstack((np.ones((1)), x.reshape(x.shape[0], -1)))
    return np.hstack((np.ones((x.shape[0], 1)), x.reshape(x.shape[0], -1)))

def add_zero_bias(x):
    if 1 == x.shape[1] or len(x.shape) == 1:
        return np.vstack((np.zeros((1)), x.reshape(x.shape[0], -1)))
    return np.hstack((np.zeros((x.shape[0], 1)), x.reshape(x.shape[0], -1)))