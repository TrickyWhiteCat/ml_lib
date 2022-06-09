from .tools import preprocess
from .tools import func
from models import Model
import numpy as np


class NN(Model):
    def __init__(self, nodes_per_layer:tuple|list):
        self._NUM_NODES = nodes_per_layer
        self._NUM_LAYERS = len(nodes_per_layer)
        
