from .tools import preprocess
from .tools import func
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='logistic_regression.log', filemode='w')

def _cost(theta, x, y, lambda_):
    SAMPLE_SIZE = np.size(x, 0)
    reg = lambda_ / (2 * SAMPLE_SIZE) * func.sum(np.square(theta))
    y_hat = func.sigmoid(x @ theta)
    c = -func.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return c + reg

def _grad(theta, x, y, lambda_):
    SAMPLE_SIZE = np.size(x, 0)
    y_hat = func.sigmoid(x @ theta)
    grad = (x.T @ (y_hat - y)) / SAMPLE_SIZE + lambda_ / SAMPLE_SIZE * theta
    return grad

def _gradient_descent(x, y, lambda_, learning_rate, iterations, **kwargs):
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

def _stochastic_gradient_descent(x, y, lambda_, learning_rate, iterations, **kwargs):
    logger = logging.getLogger('StochasticGD')
    NUM_FEATURES = x.shape[1]
    if 'plot_cost' in kwargs:
        plot = kwargs['plot_cost']
    else:
        plot = False
    theta = preprocess.init_theta(x.shape[1])
    x, y = func.shuffle(x, y)
    costs = []
    for i in range(iterations):
        logger.info('Iteration {} is started'.format(i + 1))
        temp = 0
        for j in range(x.shape[0]):
            theta -= learning_rate * _grad(theta, x[j].reshape(1, NUM_FEATURES), y[j], lambda_)
            if plot:
                # Plot the avg cost of every 100 iterations
                temp = temp + _cost(theta, x, y, lambda_)[0]
                if (j + 1) % 500 == 0 or j == x.shape[0] - 1:
                    costs =costs + [temp / 500 if (j + 1) % 100 == 0 else temp / (x.shape[0] % 500)]
                    temp = 0
        logger.info('Iteration {} is finished'.format(i + 1))

    plt.xlabel('Iteration (x1000)')
    plt.ylabel('cost')
    plt.plot(np.array(costs))
    return theta

class LogisticRegression:
    def __init__(self):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='logistic_regression.log', filemode='w')
        self._use_gd = False
        self._method = None
        self._iterate_num = 50
        self.disp = False
        self._lambda_ = 0
        self._learning_rate = 1
        logging.info('Object initialized')

    def set_x(self, value):
        self._x = value
        self._SAMPLE_SIZE = np.size(value, 0)
        self._SIZE = np.size(value)
        logging.info('x is set')

    def set_y(self, value):
        self._y = np.array(value)
        self._CLASSES = np.unique(self._y)
        self._CLASSES.sort()
        self._NUM_CLASS = np.size(self._CLASSES)
        logging.info('y is set')

    def set_lambda(self, value: float):
        self._lambda_ = value if type(value) in [float, int] else 0.1
        logging.info('lambda is set to {}'.format(value))

    def set_method(self, value):
        self._method = value
        logging.info('method is set to {}'.format(value))
    
    def set_learning_rate(self, value: float):
        self._learning_rate = value
        logging.info('learning_rate set to {}'.format(value))

    def set_iter_num(self, value):
        self._iterate_num = value
        logging.info('iterate_num is set to {}'.format(value))

    def set_scaling_method(self, value):
        if value not in ['standardize', 'normalize']:
            raise ValueError('scaling_method must be "standardize" or "normalize"')
        self._scaling_method = value
        logging.info('scaling_method is set to {}'.format(value))

    def _scale_feature_x(self):
          # Make sure x is 2D array
        x = np.reshape(np.copy(self._x), (self._SAMPLE_SIZE, self._SIZE // self._SAMPLE_SIZE))
        if self._scaling_method == 'standardize':
            self._scaling = preprocess.stardardize(x)
        else:
            self._scaling = preprocess.normalize(x)
        logging.info('Data preprocessed using {}'.format(self._scaling_method))

    def _scale_feature(self, x):
        return np.nan_to_num((x - self._scaling[1]) / self._scaling[2])

    def use_gradient_descent(self, value: bool):
        self._use_gd = value
        logging.info('use_gradient_descent is set to {}'.format(value))

    def fit(self): # Using one vs all
        logging.info('Fitting data...')

        # Ignore numpy warning
        np.seterr(all='ignore')

        # Initialize some constants and parameters
        SAMPLE_SIZE = self._SAMPLE_SIZE
        lambda_ = self._lambda_
        method = self._method
        num_iters = self._iterate_num

        # Data preprocessing
          # Feature scaling
        self._scale_feature_x()
          # Add a column of 1s to x for the bias
        x = np.hstack((np.ones((SAMPLE_SIZE, 1)), self._scaling[0]))

        # Map y to one-hot encoding to use one vs all if necessary
        y = np.array(pd.get_dummies(self._y))

        res = []
        for i in range(self._NUM_CLASS):
            y_i = y[:, i]
            if self._use_gd:
                if SAMPLE_SIZE > 1000000:
                    logging.warning('Use stochastic gradient descent for large sample size')
                    theta = _stochastic_gradient_descent(x, y_i, lambda_, learning_rate=self._learning_rate, iterations = num_iters, plot_cost=self.disp)
                else:
                    theta = _gradient_descent(x, y_i, lambda_, learning_rate= self._learning_rate, iterations=num_iters, plot_cost=self.disp)
            else:
                theta = preprocess.init_theta(x.shape[1])
                theta = opt.minimize(fun= _cost, x0= theta,args= (x, y_i, lambda_), jac = _grad, method=method, options= {'maxiter': num_iters, 'disp': self.disp}).x
            res.append(theta)
        self._theta = np.array(res)
        if self.disp:
            plt.show()
        logging.info('Model fitted')

    def predict(self, sample):
        '''Can only predict 1 sample at a time.

        Parameter:
        ---
        sample: 1D array/vector

        Return:
        ---
        Predicted label
        '''
        x = self._scale_feature(np.array(sample).reshape(1, np.size(sample)))
        x = np.hstack((np.array([1]).reshape(1, 1), x))
        res = func.softmax(func.sigmoid(x @ self._theta.T).reshape(self._NUM_CLASS, 1))
        id = np.argmax(res)
        return self._CLASSES[id]

    def cost(self):
        SAMPLE_SIZE = self._SAMPLE_SIZE
        SIZE = self._SIZE
        x = np.reshape(np.copy(self._x), (SAMPLE_SIZE, SIZE // SAMPLE_SIZE))
        theta = self._theta
        lambda_ = self._lambda_
        scaling = self._scaling
        if scaling[0] == 'standardize':
            x = (np.array(x) - scaling[1]) / scaling[2]
        elif scaling[0] == 'normalize':
            max = scaling[1]
            min = scaling[2]
            x = (np.array(x) - min) / (max - min)
        x = np.hstack((np.ones((SAMPLE_SIZE, 1)), x))        
        y = np.array(pd.get_dummies(self._y))
        res = []
        for i in range(self._NUM_CLASS):
            y_i = y[:, i]
            res += [_cost(theta[i], x, y_i, lambda_)]
        return np.array(res)


