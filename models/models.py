from tempfile import TemporaryFile
from .tools import preprocess
from .tools import func
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import numpy as np
import logging


logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='model.log', filemode='w')
class Model:
    def __init__(self):
        self._num_iters = None
        logging.info('Object initialized')
        self._scaling_method = None

    def fit(self):
        pass

    def set_x(self, value):
        self._x = value
        self._NUM_SAMPLES = np.size(value, 0)
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
        x = np.reshape(np.copy(self._x), (self._NUM_SAMPLES, self._SIZE // self._NUM_SAMPLES))
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
        for i in range(self._NUM_SAMPLES):
            if self.predict(self._x[i]) == self._y[i]:
                self._accuracy += 1
        self._accuracy /= self._NUM_SAMPLES
        return self._accuracy


        