import numpy as np


def init_theta(size):
    return np.random.randn(size)
def normalize(x):
    max = x.max()
    min = x.min()
    return (x - min) / (max - min), max, min
def stardardize(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std