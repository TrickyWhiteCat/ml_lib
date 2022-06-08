from .tools import preprocess
from .tools import func
from models.models import Model
import numpy as np

class KNN(Model):
    def __init__(self):
        self._scaling_method = None
        self._scaled = False

    def set_scaling_method(self, scaling_method):
        self._scaling_method = scaling_method
        if (not self._scaled) and self._x.any():
            self._scale_feature_x()
            self._scaled = True

    def set_k(self, value):
        try:
            self._k = int(value)
        except ValueError:
            raise ValueError('k must be an integer')

    def set_x(self, value):
        self._x = value
        self._NUM_SAMPLES = np.size(value, 0)
        self._SIZE = np.size(value)
        if (not self._scaled) and self._scaling_method:
            self._scale_feature_x()
            self._scaled = True

    def set_y(self, value):
        self._y = value
        self._CLASSES = np.unique(value)
        self._NUM_CLASSES = np.size(self._CLASSES)

    def distance(self, x):
        return np.array([np.linalg.norm(x - c) for c in self._scaling[0]])
    
    def predict(self, sample):
        if self._scaling_method:
            sample = self._scale_feature(sample.reshape(1, -1))
        distance = self.distance(sample)
        order = np.argsort(distance)
        data = [distance[order][:self._k], self._y[order][:self._k]]
        k_val = data[0]
        y = data[1]
        
        # Vote for the class
        class_count = {}
        for i in range(self._k):
            if y[i] not in class_count:
                class_count[y[i]] = np.exp(-k_val[i]**2)
            else:
                class_count[y[i]] += np.exp(-k_val[i]**2)
        return max(class_count, key=class_count.get)
        
        
