from .tools import func, preprocess
from .models import Model
import numpy as np


class NeuralNet(Model):

    def __init__(self, *num_nodes_per_layer):
        np.seterr(divide='ignore', invalid='ignore', over = 'ignore')
        self._type = 'regression'
        self._cost_func = self._mean_square
        self._NUM_NODES = tuple(num_nodes_per_layer)
        self._NUM_LAYERS = len(num_nodes_per_layer)
        self._learning_rate = 1
        self._lambda_ = 0
        # Initialize weight matrices
        self._weights = [np.array(0), np.array(0)] # np.array(0) is added so that the index will start from 1 instead of 0
        for idx in range(len(num_nodes_per_layer) - 1):
            # Add to the list a matrix with size (num_nodes_target_layer, num_nodes_prev_layers + 1). +1 for the bias.
            self._weights.append(np.random.rand(num_nodes_per_layer[idx + 1], num_nodes_per_layer[idx] + 1) - 0.5) # Just to make it have both negative and positive elements

    def set_x(self, value: np.array):
        '''Set the training set for the model. Samples must be preprocessed before added to the model. X must be a 2D array.'''

        shape = value.shape

        if len(shape) != 2:
            raise ValueError(f"X must be a 2D array while its current shape is {shape}")
        
        # Check if X fits the input layer
        if shape[1] != self._NUM_NODES[0]:
            raise ValueError(f"The number of features doesn't fit the input layer. Expected {self._NUM_NODES[0]}, got {shape[1]}")

        self._NUM_SAMPLES = shape[0]
        
        self._x = value

    def set_y(self, value: np.array):
        '''Set the ground-truth value. Y must be either 1D array or 2D array containing the ground-truth value of all samples'''
        shape = value.shape
        
        # Shape checking
        if len(shape) > 2:
            raise ValueError(f"Y must either a 1D or 2D array while its current shape is {shape}")

        # Reshape Y into m x n 2D array where m is the number of samples and n is the number of output nodes.
        value = value.reshape(self._NUM_SAMPLES, -1)

        # The length of Y's 2nd dimension must equal to the number of nodes of the output layer
        if value.shape[-1] != self._NUM_NODES[-1]:
            raise ValueError(f"Y's 2nd dimension's length({value.shape[-1]}) and the number of output nodes({self._NUM_NODES[-1]}) are mismatched")

        self._y = value

    def set_learning_rate(self, value):
        self._learning_rate = float(value)

    def set_lambda(self, value):
        self._lambda_ = float(value)

    @property
    def hidden_activation_function(self):
        return self._activation_function['name']
    @hidden_activation_function.setter
    def hidden_activation_function(self, value):
        '''Activation function of hidden layers'''
        self._hidden_activation_function = func.get_act_func(value)

    @property
    def output_activation_function(self):
        return self._output_activation_function['name']
    @output_activation_function.setter
    def output_activation_function(self, value):
        self._output_activation_function = func.get_act_func(value)

    @property
    def type(self):
        return self._type
    @type.setter
    def type(self):
        accept_type = ('regression', 'classification')
        if type not in accept_type:
            raise ValueError(f'{type} is not supported. Expected: {accept_type}')
        self._type = type
        self._cost_func = self._cross_entropy if type == accept_type[1] else self._mean_square
        

    def _fwd(self, sample):
        '''Samples must be preprocessed before added to the model and have the shape of (n, 1)'''

        hidden_act_func = self._hidden_activation_function['act_func']
        hidden_deri = self._hidden_activation_function['deri']
        output_act_func = self._output_activation_function['act_func']
        output_deri = self._output_activation_function['deri']

        sample = sample.reshape(-1, 1)

        a = [np.array(0), sample] # sample == a[1]
        grad = [np.array(0), np.array(0)]
        for w in self._weights[2:]:
            z = w @ preprocess.add_bias(a[-1])
            grad.append(hidden_deri(z))
            a.append(hidden_act_func(z))
        a[-1] = output_act_func(z).reshape(-1, 1)
        grad[-1] = output_deri(z).reshape(-1, 1)

        return a, grad

    def _cost(self, sample, ground_truth):
        y = self._fwd(sample)[0][-1]
        reg = 0
        for w in self._weights:
            reg += self._lambda_ * np.sum(w**2) / 2
        return -(ground_truth.T @ np.log(y)) -((1 - ground_truth.T) @ np.log(1-y)) + reg

    def _back_prop(self, sample, y):
        '''Return the derivative of the cost function w.r.t the weights using ONLY ONE SAMPLE'''
        
        #
        a, g = self._fwd(sample)
        # Cost function: cross entropy
        y = y.reshape(-1, 1)
        delta = self._cost_func(y = a[-1].reshape(-1, 1), ground_truth=y)[1]
        grad = []
        for idx in range(self._NUM_LAYERS, 1, -1):
            grad.append(delta @ preprocess.add_bias(a[idx - 1]).T + self._lambda_ * self._weights[idx])
            delta = g[idx - 1] * (self._weights[idx].T @ delta)[1:]

        return grad[::-1]

    def _update_weights(self, grad):
        for idx in range(self._NUM_LAYERS, 1, -1):
            self._weights[idx] -= self._learning_rate * grad[idx -2]

    def fit(self):
        for sample in zip(self._x, self._y):
            #print(self._cost(sample[0], sample[1]))
            grad = self._back_prop(sample[0], sample[1])
            self._update_weights(grad)

    def predict(self, sample):
        return np.argmax(self._fwd(sample.reshape(1, -1))[0][-1])

    # Cost functions; return the value of the cost and its derivative w.r.t its input

    def _mean_square(self, y, ground_truth):
        '''Return: cost, delta'''
        reg = 0
        reg_d = 0
        for w in self._weights:
            reg += self._lambda_ * np.sum(w**2) / 2
            reg_d += self._lambda_ * np.sum(w) / 2
        diff = (ground_truth - y).reshape(-1, 1)
        return (diff @ diff.T + reg) / 2, diff

    def _cross_entropy(self, y, ground_truth):
        '''Return: cost, delta'''
        reg = 0
        for w in self._weights:
            reg += self._lambda_ * np.sum(w**2) / 2
        return (-(ground_truth.T @ np.log(y)) -((1 - ground_truth.T) @ np.log(1-y)) + reg) / y.shape[0], ground_truth - y