from .tools import preprocess
from .tools import func
from models import Model
import numpy as np


class NN(Model):
    def __init__(self, nodes_per_layer:tuple|list):
        self._NUM_NODES = nodes_per_layer
        self._NUM_LAYERS = len(nodes_per_layer)
<<<<<<< HEAD
        self._weights = []
        for i in range(self._NUM_LAYERS - 1):
            self._weights.append(np.random.randn(self._NUM_NODES[i]+1, self._NUM_NODES[i + 1]+1)) # +1 for bias

    def set_x(self, value):
        if value.ndim == 1:
            value = value.reshape(1, -1)
            if value.shape[1] != self._NUM_NODES[0]:
                raise ValueError("The input vector has the wrong dimension.")
        elif value.ndim == 2:
            if value.shape[1] != self._NUM_NODES[0]:
                raise ValueError("The input matrix has the wrong dimension.")
        else:
            raise ValueError("The input has the wrong dimension: expected 1 or 2, got {}.".format(value.ndim))
        self._x = value

    def set_y(self, value):
        if value.ndim == 1:
            value = value.reshape(1, -1)
            if value.shape[1] != self._NUM_NODES[-1]:
                raise ValueError("The output has the wrong dimension.")
        elif value.ndim == 2:
            if value.shape[1] != self._NUM_NODES[-1]:
                raise ValueError("The output has the wrong dimension.")
        else:
            raise ValueError("The output has the wrong dimension: expected 1 or 2, got {}.".format(value.ndim))
        self._y = value

    @property
    def activation_func(self):
        return self._activation_func['name']
    @activation_func.setter
    def activation_func(self, value):
        self._activation_func = func.get_act_func(value)

    def _forward_prop(self, sample):
        '''
        Forward propagation of the neural network.
        ---
        Parameters:
        sample: numpy.ndarray: the input sample
        ---
        Returns:
        z: 
        '''
        act_func = self._activation_func['act_func']
        grad = self._activation_func['grad']

        temp = preprocess.add_bias(sample)
        z = [temp]
        a = [temp]
        for i in range(self._NUM_LAYERS - 1):
            z.append(self._weights[i].T @ z[i])
            a.append(act_func(z[i+1]))
        
        return z, a
        