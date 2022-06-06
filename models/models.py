from .tools import preprocess
from .tools import func
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='model.log', filemode='w')
class Model:
    def __init__(self):
        self._num_iters = None
        logging.info('Object initialized')
        self._scaling_method = None

    def fit(self):
        pass

    def set_x(self, value):
        self._x = value
        self._SAMPLE_SIZE = np.size(value, 0)
        self._SIZE = np.size(value)
        logging.info('x is set')

    def set_num_iters(self, value):
        self._num_iters = value
        logging.info('num_iters is set to {}'.format(value))

    def set_scaling_method(self, value):
        if value not in ['standardize', 'normalize']:
            raise ValueError('scaling_method must be "standardize" or "normalize"')
        self._scaling_method = value

    def _scale_feature_x(self):
          # Make sure x is 2D array
        x = np.reshape(np.copy(self._x), (self._SAMPLE_SIZE, self._SIZE // self._SAMPLE_SIZE))
        if not self._scaling_method: return
        if self._scaling_method == 'standardize':
            self._scaling = preprocess.stardardize(x)
        elif self._scaling_method == 'normalize':
            self._scaling = preprocess.normalize(x)

    def _scale_feature(self, x):
        if not self._scaling_method: return x
        return np.nan_to_num((x - self._scaling[1]) / self._scaling[2])

    def fit(self):
        pass

    def predict(self, sample):
        pass

    def train_accuracy(self):
        self._accuracy = 0
        for i in range(self._SAMPLE_SIZE):
            if self.predict(self._x[i]) == self._y[i]:
                self._accuracy += 1
        self._accuracy /= self._SAMPLE_SIZE
        return self._accuracy


class KMeans(Model):
    def __init__(self):
        self._centroids = []
        self._num_iters = 10
        self._num_centroids = 2
        self._scaling_method = None

    def set_num_centroids(self, value):
        self._num_centroids = value
        logging.info('num_centroids is set to {}'.format(value))

    def set_num_iters(self, value):
        '''
        Set the number of times that the algorithm will dependently run. For K-means, this is also the number of times that centroids are randomly chosen. The one with the least cost will be returned.
        '''
        self._num_iters = value
        logging.info('num_iters is set to {}'.format(value))

    @property
    def centroids(self):
        if not self._scaling_method: return self._centroids
        return self._centroids * self._scaling[2] + self._scaling[1]

    def _init_centroids(self):
        x = self._scaling[0]
        c = []
        for i in range(self._num_centroids):
            c.append(x[np.random.randint(0, self._SAMPLE_SIZE)])
        self._centroids = np.array(c)

    def _distance_to_centroids(self, x):
        return np.array([np.linalg.norm(x - c) for c in self._centroids])

    def _cost(self):
        return np.sum(self._distance_to_centroids(self._scaling[0]))

    def _assign(self):
        x = self._scaling[0]
        self._y = np.zeros(self._SAMPLE_SIZE)
        for i in range(self._SAMPLE_SIZE):
            self._y[i] = np.argmin(self._distance_to_centroids(x[i]))

    def _update(self):
        x = self._scaling[0]
        for i in range(self._num_centroids):
            self._centroids[i] = np.mean(x[self._y == i], axis=0)

    def _converged(self):
        return np.allclose(self._centroids, self._prev)

    def fit(self):
        # Scale features
        self._scale_feature_x()

        # Initialize centroids
        self._init_centroids()

        while True:
            # Assign to clusters
            self._assign()

            # Update centroids
            self._prev = np.copy(self._centroids)
            self._update()

            # Check if converged
            if self._converged():
                break


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
        SAMPLE_SIZE = self._SAMPLE_SIZE
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


