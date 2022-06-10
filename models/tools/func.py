import numpy as np
import matplotlib.pyplot as plt
import preprocess


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
        return 1 / (1 + np.exp(-x))
    def grad_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))
    def relu(x):
        return np.maximum(0, x)
    def grad_relu(x):
        return 0 if x < 0 else 1
    act_fun = act_fun.lower()
    if act_fun == 'sigmoid':
        return {'act_func': sigmoid, 'grad': grad_sigmoid, 'name': 'sigmoid'}
    if act_fun == 'relu':
        return {'act_func': relu, 'grad': grad_relu, 'name': 'relu'}
    raise ValueError('Invalid activation function')