from .tools import preprocess
from .tools import func
import matplotlib.pyplot as plt
from scipy import optimize as opt
import numpy as np


class Model:
    def __init__(self):
        self._num_iters = 50
        self._scaling_method = None

    def set_x(self, value):
        self._x = value
        self._NUM_SAMPLES = np.size(value, 0)
        self._SIZE = np.size(value)

    def set_num_iters(self, value):
        self._num_iters = value

    def set_scaling_method(self, value):
        if value not in ['standardize', 'normalize']:
            raise ValueError('scaling_method must be "standardize" or "normalize"')
        self._scaling_method = value

    def _scale_feature_x(self):
          # Make sure x is 2D array
        x = np.reshape(np.copy(self._x), (self._NUM_SAMPLES, self._SIZE // self._NUM_SAMPLES))
        if not self._scaling_method: 
            self._scaling =  np.array(x), np.array(0), np.array(1)
        if self._scaling_method == 'standardize':
            self._scaling = preprocess.standardize(x)
        elif self._scaling_method == 'normalize':
            self._scaling = preprocess.normalize(x)

    def _scale_feature(self, x):
        if not self._scaling_method: return np.array(x)
        return np.nan_to_num((x - self._scaling[1]) / self._scaling[2])

    def fit(self):
        pass

    def predict(self, sample):
        pass

    def train_accuracy(self):
        try: 
            if not self._y:
                raise AttributeError('y is not set')
        except:
            self._accuracy = 0
            for i in range(self._NUM_SAMPLES):
                if self.predict(self._x[i]) == self._y[i]:
                    self._accuracy += 1
            self._accuracy /= self._NUM_SAMPLES
            return self._accuracy


        