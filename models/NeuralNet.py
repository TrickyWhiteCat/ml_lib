from .models import Model
import numpy as np

import matplotlib.pyplot as plt

class NeuralNet(Model):

    def __init__(self, *num_nodes_per_layer):
        self.NUM_LAYERS = len(num_nodes_per_layer)
        