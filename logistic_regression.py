import preprocess
import func
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='logistic_regression.log', filemode='w')


def _cost(theta, x, y, lambda_):
    SAMPLE_SIZE = np.size(x, 0)
    reg = lambda_ / (2 * SAMPLE_SIZE) * np.sum(np.square(theta))
    y_hat = func.sigmoid(x @ theta)
    c = -func.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return c + reg

def _grad(theta, x, y, lambda_):
    SAMPLE_SIZE = np.size(x, 0)
    y_hat = func.sigmoid(x @ theta)
    grad = (x.T @ (y_hat - y)) / SAMPLE_SIZE + lambda_ / SAMPLE_SIZE * theta
    return grad

def _gradient_decent(x, y, lambda_, learning_rate, iterations, **kwargs):
    if 'plot_cost' in kwargs:
        plot = kwargs['plot_cost']
    else:
        plot = False
    theta = preprocess.init_theta(x.shape[1])
    for i in range(iterations):
        theta -= learning_rate * _grad(theta, x, y, lambda_)
        if plot:
            cost = _cost(theta, x, y, lambda_)
            costs = [] if i == 0 else costs + [cost]
    if plot:
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.plot(costs)
    return theta

class LogisticRegression:
    def __init__(self):
        self._use_gd = False
        self._method = None
        self._iterate_num = None
        logging.info('Object initialized')

    def set_x(self, value):
        self._x = value
        self._SAMPLE_SIZE = np.size(value, 0)
        self._SIZE = np.size(value)
        logging.info('x set')

    def set_y(self, value):
        self._y = np.array(value)
        self._NUM_CLASS = np.size(np.unique(self._y))
        logging.info('y set')

    def set_lambda(self, value: float):
        self._lambda_ = value if type(value) in [float, int] else 0.1
        logging.info('lambda set to {}'.format(value))

    def set_method(self, value):
        self._method = value
        logging.info('method set to {}'.format(value))

    def set_iter_num(self, value):
        self._iterate_num = value
        logging.info('iterate_num set to {}'.format(value))

    def set_scaling_method(self, value):
        if value not in ['standardize', 'normalize']:
            raise ValueError('scaling_method must be "standardize" or "normalize"')
        self._scaling_method = value
        logging.info('scaling_method set to {}'.format(value))

    def use_gradient_descent(self, value: bool):
        self._use_gd = value
        logging.info('use_gradient_descent set to {}'.format(value))

    def fit(self): # Using one vs all
        logging.info('Fitting data...')

        np.seterr(all='ignore')

        SAMPLE_SIZE = self._SAMPLE_SIZE
        SIZE = self._SIZE
        lambda_ = self._lambda_
        method = self._method
        num_iters = self._iterate_num
        scaling_method = self._scaling_method
        # Data preprocessing
        x = np.reshape(np.copy(self._x), (SAMPLE_SIZE, SIZE // SAMPLE_SIZE))
    #x.reshape(SAMPLE_SIZE, SIZE // SAMPLE_SIZE) # Make sure x is a 2D array
        if scaling_method is not None:
            if scaling_method == 'standardize':
                x, mean, std = preprocess.stardardize(x)
                self._scaling = (scaling_method, mean, std)
            if scaling_method == 'normalize':
                x, max, min = preprocess.normalize(x)
                self._scaling = (scaling_method, max, min)

        x = np.hstack((np.ones((SAMPLE_SIZE, 1)), x))
        logging.info('Data preprocessed using {}'.format(self._scaling[0]))

        y = np.array(pd.get_dummies(self._y))

        res = []
        for i in range(y.shape[1]):
            y_i = y[:, i]
            if self._use_gd:
                theta = _gradient_decent(x, y_i, lambda_, 0.1, iterations=num_iters)
                logging.info('Gradient descent has been used {} times'.format(i + 1))
            else:
                theta = preprocess.init_theta(x.shape[1])
                theta = opt.minimize(fun= _cost, x0= theta,args= (x, y_i, lambda_), jac = _grad, method=method, options= {'maxiter': num_iters}).x
            res.append(theta)
        self._theta = np.array(res)
        logging.info('Model fitted')

    def predict(self, sample):
        '''Can only predict 1 sample at a time'''
        scaling = self._scaling
        if scaling[0] == 'standardize':
            x = (np.array(sample) - scaling[1]) / scaling[2]
        elif scaling[0] == 'normalize':
            max = scaling[1]
            min = scaling[2]
            x = (np.array(sample) - min) / (max - min)
        else:
            x = np.array(sample)
        x = np.hstack((np.array([1]).reshape(1, 1), np.reshape(x, (1, np.size(x)))))
        return np.argmax(func.sigmoid(x @ self._theta.T))


if __name__ == '__main__':
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    lr = LogisticRegression()
    lr.set_x(x_train[:10000])
    lr.set_y(y_train[:10000])
    lr.use_gradient_descent(False)
    lr.set_lambda(0.1)
    lr.set_iter_num(50)
    lr.set_method('BFGS')
    lr.set_scaling_method('standardize')
    lr.fit()
    correct = 0
    for i in range(np.size(x_test, 0)):
        correct += lr.predict(x_test[i]) == y_test[i]
    print('accuracy: ', correct / np.size(x_test, 0))