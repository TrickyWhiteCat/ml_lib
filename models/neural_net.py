import numpy as np
import pandas as pd
from .models import Model
from .tools import preprocess
from .tools import func


class NN(Model):
    def __init__(self, *nodes_per_hidden_layer):
        '''
        Initialize a neural network with the given number of nodes per layer.
        '''
        super().__init__()
        self._NUM_NODES = list(nodes_per_hidden_layer)
        self._NUM_LAYERS = len(nodes_per_hidden_layer) +2 # +1 for input layer, +1 for output layer
        self._weights = []
        self._epochs = 50
        self._learning_rate = 0.01
        self._activation_func = func.get_act_func('sigmoid')
        self._theta_initialized = False
        
    def _init_theta(self):
        '''
        Initializes the weights of the neural network.
        '''
        try:
            if self._theta_initialized or not (self._x.any() and self._y.any()):
                return
        except AttributeError:
            return
        w = []
        for i in range(self._NUM_LAYERS - 1):
            # No bias needed for output layer
            w.append(np.random.randn(self._NUM_NODES[i]+1, self._NUM_NODES[i + 1]))
        self._weights = w
        self._theta_initialized = True

    def set_x(self, value):
        value = np.array(value)
        # Make sure the input is a 2D array
        value = value.reshape(value.shape[0], -1)
        if value.ndim not in (1, 2):
            raise ValueError(f"The input has the wrong dimension: expected 1 or 2, got {value.ndim}.")
        self._x = value
        self._SIZE = np.size(value)
        self._NUM_SAMPLES = value.shape[0]
        # Add input layer to the front of the list of weights
        self._NUM_NODES.insert(0, value.shape[1])
        self._init_theta()

    def set_y(self, value):
        value = np.array(pd.get_dummies(value))
        if value.ndim not in (1, 2):
            raise ValueError(f"The output has the wrong dimension: expected 1 or 2, got {value.ndim}.")
        self._y = value.reshape(value.shape[0], -1)
        # Add output layer to the end of the list of weights
        self._NUM_NODES.append(self._y.shape[1])
        self._init_theta()

    def set_epochs(self, value):
        try:
            self._epochs = int(value)
        except ValueError:
            raise ValueError("The number of epochs must be an integer.")

    def set_learning_rate(self, value):
        try:
            self._learning_rate = float(value)
        except ValueError:
            raise ValueError("The learning rate must be a float or int.")

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
        a: numpy.ndarray: the output of the neural network stored in a list of arrays
        g: numpy.ndarray: the gradient of the output of the neural network stored in a list of arrays
        '''
        act_func = self._activation_func['act_func']
        grad = self._activation_func['grad']

        z = sample.reshape(sample.shape[0], 1)
        a = [z]
        g = []
        for i in range(self._NUM_LAYERS - 1):
            z = preprocess.add_bias(a[-1])
            g.append(grad(z))
            z = self._weights[i].T @ z # (nodes_per_layer[i+1], nodes_per_layer[i]+1) @ (nodes_per_layer[i]+1, 1) = (nodes_per_layer[i+1], 1)
            a.append(act_func(z))
        g.append(grad(z))
        return a, g
        
    def _backward_prop(self, sample, y):
        '''
        Backward propagation of the neural network.
        ---
        Parameters:
        sample: numpy.ndarray: the input sample
        y: numpy.ndarray: the expected output
        ---
        Returns:
        grad: numpy.ndarray: the gradient of the cost function with respect to the weights
        '''
        
        # Performing feedforward
        a, g = self._forward_prop(sample)

        # Computing the gradient of the cost function
        grad_cost = []
        for i in range(self._NUM_LAYERS - 1):
            grad_cost.append(np.zeros((self._NUM_NODES[i] + 1, self._NUM_NODES[i + 1])))
        grad_cost.append(g[-1] * (a[-1] - y.reshape(y.shape[0], 1)))
        for i in range(self._NUM_LAYERS - 2, -1, -1):
            grad_cost[i] = g[i] * (self._weights[i] @ grad_cost[i + 1])
        return grad_cost

    def _flatten_x(self, x):
        flat = []
        original_shape = []
        if type(x) == 'list' or type(x) == 'tuple':
            for i in range(len(x)):
                original_shape.append(x[i].shape)
                flat.extend(x[i].flatten())
            flat = np.array(flat)
            return flat, original_shape
            
        if type(x) == 'np.ndarray':
            original_shape.append(x.shape)
            return x.flatten(), original_shape
        else:
            raise ValueError(f'Invalid type. Require: list, tuple, np.ndarray while got {type(x)}')

    def _recover_grad(self, flat, original_shape):
        grad = []
        # Internal function to recover the gradient. No need to check for the type of parameters
        for i in range(len(original_shape)):
            grad.append(flat[:original_shape[i][0]*original_shape[i][1]].reshape(original_shape[i]))
            flat = flat[original_shape[i][0]*original_shape[i][1]:]
        return grad

    def _update_weights(self, grad):
        '''
        Updates the weights of the neural network.
        ---
        Parameters:
        grad: numpy.ndarray: the gradient of the cost function with respect to the weights
        ---
        Returns:
        None
        '''
        for i in range(self._NUM_LAYERS - 1):
            self._weights[i] -= self._learning_rate * grad[i]
    
    def fit(self):
        '''
        Fits the neural network to the data.
        '''
        self._scale_feature_x()
        x = self._scaling[0]
        y = self._y
        for i in range(self._epochs):
            for j in range(x.shape[0]):
                grad = self._backward_prop(x[j], y[j])
                self._update_weights(grad)
            print(f"Epoch {i+1}")

    def predict(self, sample):
        '''
        Predicts the output of the neural network.
        ---
        Parameters:
        x: numpy.ndarray: the input data
        ---
        Returns:
        y: numpy.ndarray: the predicted output
        '''
        a, _ = self._forward_prop(sample.reshape(np.size(sample), 1))
        return np.argmax(func.softmax(a[-1]))
