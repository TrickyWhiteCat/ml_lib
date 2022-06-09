from .tools import preprocess
from .tools import func
from models import Model
import numpy as np


class NN(Model):
    def __init__(self, nodes_per_layer:tuple|list):
        self._NUM_NODES = nodes_per_layer
        self._NUM_LAYERS = len(nodes_per_layer)
        for i in range(self._NUM_LAYERS - 1):
            self._weights.append(np.random.rand(self._NUM_NODES[i], self._NUM_NODES[i+1]))
