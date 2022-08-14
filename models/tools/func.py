import numpy as np
import matplotlib.pyplot as plt
from . import preprocess


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

def shuffle(x: np.ndarray, y: np.ndarray):
    '''
    Return suffled data
    '''
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index], y[index]

def gradient_descent(x, y, lambda_, grad, learning_rate, iterations, costf = None, **kwargs):
    if 'plot_cost' in kwargs:
        plot = kwargs['plot_cost']
    else:
        plot = False
    theta = preprocess.init_theta(x.shape[1])
    for i in range(iterations):
        theta -= learning_rate * grad(theta, x, y, lambda_)
        if plot and costf:
            cost = costf(theta, x, y, lambda_)
            costs = [] if i == 0 else costs + [cost]
    if plot and costf:
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.plot(costs)
        plt.show()
    return theta

def stochastic_gradient_descent(x, y, lambda_, grad, learning_rate, iterations, costf = None, **kwargs):
    NUM_FEATURES = x.shape[1]
    if 'plot_cost' in kwargs:
        plot = kwargs['plot_cost']
    else:
        plot = False
    theta = preprocess.init_theta(x.shape[1])
    x, y = shuffle(x, y)
    costs = []
    for i in range(iterations):
        temp = 0
        for j in range(x.shape[0]):
            theta -= learning_rate * grad(theta, x[j].reshape(1, NUM_FEATURES), y[j], lambda_)
            if plot and costf:
                # Plot the avg cost of every 100 iterations
                temp = temp + np.nan_to_num(costf(theta, x, y, lambda_))[0] 
                if (j + 1) % 100 == 0 or j == x.shape[0] - 1:
                    costs =costs + [temp / 100 if (j + 1) % 100 == 0 else temp / (x.shape[0] % 100)]
                    temp = 0
    if plot and costf:
        plt.xlabel('Iteration (x100)')
        plt.ylabel('cost')
        plt.plot(np.array(costs))
        plt.show()
    return theta

def get_act_func(act_fun):

    def sigmoid(x):
        return np.nan_to_num(1 / (1 + np.exp(-x)))
    def deri_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def relu(x):
        return np.maximum(0, x)
    def deri_relu(x):
        return 1 * (x > 0)

    def leaky_relu(x):
        return np.where(x > 0, x, x * 0.001)
    def deri_leaky_relu(x):
        return np.where(x > 0, 1, 0.001)

    def softmax(x):
        e_x = np.exp(x)
        A = np.nan_to_num(e_x / e_x.sum(axis = 0))
        return A 
    def deri_softmax(x):
        return softmax(x)*(1 - softmax(x))

    supported = {
        'sigmoid': {'act_func': sigmoid, 'deri': deri_sigmoid, 'name': 'sigmoid'},
        'relu': {'act_func': relu, 'deri': deri_relu, 'name': 'relu'},
        'softmax': {'act_func': softmax, 'deri': deri_softmax, 'name': 'softmax'},
        'leaky_relu': {'act_func': leaky_relu, 'deri': deri_leaky_relu, 'name': 'leaky relu'}}

    if act_fun is None:
        return {'act_func': lambda x:x, 'deri': lambda x: np.array(1), 'name': None}

    act_fun = act_fun.lower()
    try:
        return supported[act_fun]
    except KeyError:
        raise ValueError('Unsupported activation function')

def flatten(x):
    if type(x) == 'list' or type(x) == 'tuple':
        return [flatten(i) for i in x]
    if type(x) == 'np.ndarray':
        return x.flatten()
    else:
        raise ValueError(f'Invalid type. Require: list, tuple, np.ndarray while got {type(x)}')