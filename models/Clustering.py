import numpy as np
import matplotlib.pyplot as plt
from .model import Model
from .tools import func
from .tools import preprocess
import logging


logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='kmeans.log', filemode='w')
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
    
