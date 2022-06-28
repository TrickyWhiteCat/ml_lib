from .models import Model
from .tools import func, preprocess
import numpy as np
import logging
import pandas as pd
from scipy import optimize as opt


class LogisticRegression(Model):

    def _cost(self, theta, x, y, lambda_):
        SAMPLE_SIZE = np.size(x, 0)
        reg = lambda_ / (2 * SAMPLE_SIZE) * func.sum(np.square(theta))
        y_hat = func.sigmoid(x @ theta)
        c = -func.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return c + reg

    def _grad(self, theta, x, y, lambda_):
        SAMPLE_SIZE = np.size(x, 0)
        y_hat = func.sigmoid(x @ theta)
        grad = (x.T @ (y_hat - y)) / SAMPLE_SIZE + lambda_ / SAMPLE_SIZE * theta
        return grad

    def __init__(self):
        self._use_gd = False
        self._method = None
        self.disp = False
        self._lambda_ = 0
        self._learning_rate = 1
        self._scaling_method = None
        logging.info('Object initialized')

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

    def set_num_iters(self, value):
        self._num_iters = value
        logging.info('num_iters is set to {}'.format(value))

    def fit(self): # Using one vs all
        logging.info('Fitting data...')

        # Ignore numpy warning
        np.seterr(all='ignore')

        # Initialize some constants and parameters
        SAMPLE_SIZE = self._NUM_SAMPLES
        lambda_ = self._lambda_
        method = self._method
        num_iters = self._num_iters

        # Data preprocessing
          # Feature scaling
        self._scale_feature_x()
          # Add a column of 1s to x for the bias
        x = np.hstack((np.ones((SAMPLE_SIZE, 1)), self._scaling[0]))

        # Map y to one-hot encoding to use one vs all if necessary
        y = np.array(pd.get_dummies(self._y))

        logging.info('Attempting to fit data using {}...'.format('gradient descent' if self._use_gd else ('scipy minimize with ' + method) if method else 'scipy minimize'))
        res = []
        for i in range(self._NUM_CLASS):
            y_i = y[:, i]
            if self._use_gd:
                if SAMPLE_SIZE > 1000000:
                    logging.warning('Use stochastic gradient descent for large sample size')
                    theta = func.stochastic_gradient_descent(x, y_i, lambda_, grad = self._grad, costf = self._cost, learning_rate=self._learning_rate, iterations = num_iters, plot_cost=self.disp)
                else:
                    theta = func.gradient_descent(x, y_i, lambda_, grad = self._grad, costf = self._cost, learning_rate= self._learning_rate, iterations=num_iters, plot_cost=self.disp)
            else:
                theta = preprocess.init_theta(x.shape[1])
                theta = opt.minimize(fun= self._cost, x0= theta,args= (x, y_i, lambda_), jac = self._grad, method=method, options= {'maxiter': num_iters, 'disp': self.disp}).x
            res.append(theta)
        self._theta = np.array(res)
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
