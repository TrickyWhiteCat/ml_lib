import preprocess
import numpy as np

class LogisticRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._use_gradient_descent = False

    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        self._theta = value

    @property
    def scaling_method(self):
        return self._scaling_method
    @scaling_method.setter
    def scaling_method(self, value):
        if value not in ['standardize', 'normalize']:
            raise ValueError('scaling_method must be "standardize" or "normalize"')
        self._scaling_method = value

    def use_gradient_descent(self, value: bool):
        self._use_gradient_descent = value

    def fit(self, **kwargs):
        if self.scaling_method == 'standardize':
            self.x, self.mean, self.std = preprocess.stardardize(self.x)
        elif self.scaling_method == 'normalize':
            self.x, self.max, self.min = preprocess.normalize(self.x)
        self.theta = preprocess.init_theta(self.x.shape[1])
