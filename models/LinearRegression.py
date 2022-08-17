import scipy.optimize as opt
import numpy as np
from .tools import func, preprocess
from .models import Model


class LinearRegression(Model):

    def __init__(self):
        self.use_gd = False
        self._method = None
        self.disp = False
        self._lambda_ = 0
        self._learning_rate = 1
        self._scaling_method = None
        self._num_iters = 50

    def set_y(self, value):
        self._y = np.array(value).reshape(-1, 1) # Map from a vector X into a value Y

    def _cost(self, theta, x, y, lambda_):
        theta = theta.reshape(-1,1)
        SAMPLE_SIZE = np.size(x, 0)
        reg = lambda_ / (2 * SAMPLE_SIZE) * func.sum(np.square(theta))
        y_hat = x @ theta
        diff = (y - y_hat.reshape(y.shape)).reshape(-1, 1)
        c = diff.T @ diff / (2 * SAMPLE_SIZE)
        return c + reg

    def _grad(self, theta, x, y, lambda_):
        SAMPLE_SIZE = np.size(x, 0)

        y_hat = x @ theta.reshape(-1, 1)
        grad = (x.T @ (y_hat - y)) / SAMPLE_SIZE + lambda_ / SAMPLE_SIZE * theta.reshape(-1, 1)
        return grad.reshape(-1)

    def set_method(self, value):
        self._method = value
    
    def set_learning_rate(self, value):
        self._learning_rate = value

    def set_num_iters(self, value):
        self._num_iters = value

    def fit(self):

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
        y = self._y

        if self.use_gd:
            theta = func.gradient_descent(x, y, lambda_, grad = self._grad, costf = self._cost, learning_rate= self._learning_rate, iterations=num_iters, plot_cost=self.disp)
        else:
            theta = preprocess.init_theta(x.shape[1])
            theta = opt.minimize(fun= self._cost, x0= theta,args= (x, y, lambda_), jac = self._grad, method=method, options= {'maxiter': num_iters, 'disp': self.disp}).x

        self._theta = theta

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
        x = np.hstack((np.array([1]).reshape(1, 1), x)).T
        return self._theta.T @ x