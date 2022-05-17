import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x)) if type(x) == np.ndarray else 1 / (1 + np.exp(-np.array(x)))

def sum(x: np.ndarray) -> np.ndarray:
    '''
    Return the sum of an array along the 1st dimention
    '''
    if type(x) != np.ndarray:
        x = np.array(x)
    return np.ones(shape = (1, np.size(x, axis = 0))) @ x

def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Return the softmax of an array
    '''
    if type(x) != np.ndarray:
        x = np.array(x)
    e_x = np.exp(x)
    A = e_x / e_x.sum(axis = 0)
    return A 