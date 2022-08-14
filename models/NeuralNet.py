from .tools import func, preprocess
from .models import Model
import numpy as np
import pandas as pd

try:
    import cupy as cp
    CAN_USE_CUPY = True
except ModuleNotFoundError:
    CAN_USE_CUPY = False

class NeuralNet(Model):

    def __init__(self, *num_nodes_per_layer, use_cupy = False):
        self._cp = False
        if use_cupy and CAN_USE_CUPY:
            self._cp = True
        np.seterr(divide='ignore', invalid='ignore', over = 'ignore')
        self.num_epochs = 1
        self._type = 'regression'
        self._cost_func = self._mean_square
        self._NUM_NODES = tuple(num_nodes_per_layer)
        self._NUM_LAYERS = len(num_nodes_per_layer)
        self._learning_rate = 1
        self._lambda_ = 0
        self._scaling_method = None

        # Initialize weight matrices
        self._zero = np.array(0) if not use_cupy else cp.array(0)
        self._weights = [self._zero, self._zero] # cp.array(0) is added so that the index will start from 1 instead of 0
        for idx in range(len(num_nodes_per_layer) - 1):
            # Add to the list a matrix with size (num_nodes_target_layer, num_nodes_prev_layers + 1). +1 for the bias.
            if use_cupy:
                rand_w = cp.random.rand(num_nodes_per_layer[idx + 1], num_nodes_per_layer[idx] + 1) - 0.5
            else:
                rand_w = np.random.rand(num_nodes_per_layer[idx + 1], num_nodes_per_layer[idx] + 1) - 0.5
            self._weights.append(rand_w) # Just to make it have both negative and positive elements

    def set_x(self, value):
        value = np.array(value)
        '''Set the training set for the model. X must be a 2D array.'''
        shape = value.shape

        if len(shape) != 2:
            raise ValueError(f"X must be a 2D array while its current shape is {shape}")
        
        # Check if X fits the input layer
        if shape[1] != self._NUM_NODES[0]:
            raise ValueError(f"The number of features doesn't fit the input layer. Expected {self._NUM_NODES[0]}, got {shape[1]}")

        self._NUM_SAMPLES = shape[0]
        self._SIZE = np.size(value)
        self._x = value

    def set_y(self, value):
        '''Set the ground-truth value. Y must be either 1D array or 2D array containing the ground-truth value of all samples. If NN's type is classification then type need to be set before you call this function'''

        # If type == classification, get the labels vector and dummies matrix
        if self._type == 'classification':
            dummies = pd.get_dummies(value)
            self._label = dummies.columns
            self._y = cp.array(dummies) if self._cp else np.array(dummies)
        else: 
            
            shape = value.shape
        
            # Shape checking
            if len(shape) > 2:
                raise ValueError(f"Y must either a 1D or 2D array while its current shape is {shape}")
            # Reshape Y into m x n 2D array where m is the number of samples and n is the number of output nodes.
            value = value.reshape(self._NUM_SAMPLES, -1)

            # The length of Y's 2nd dimension must equal to the number of nodes of the output layer
            if value.shape[-1] != self._NUM_NODES[-1]:
                raise ValueError(f"Y's 2nd dimension's length({value.shape[-1]}) and the number of output nodes({self._NUM_NODES[-1]}) are mismatched")

            self._y = cp.array(value) if self.cp else np.array(value)

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
    def type(self, value):
        accept_type = ('regression', 'classification')
        if value not in accept_type:
            raise ValueError(f'{value} is not supported. Expected: {accept_type}')
        self._type = value
        self._cost_func = self._cross_entropy if value == accept_type[1] else self._mean_square
        
    # Override super()'s method to convert the result to cupy.ndarray

    def _scale_feature_x(self):
        super()._scale_feature_x()
        if self._cp:
            self._scaling = [cp.array(val) for val in self._scaling]

    def _scale_feature(self, x):
        return cp.array(super()._scale_feature(x)) if self._cp else super()._scale_feature(x)

    def _fwd(self, sample):
        '''Samples must be preprocessed before added to the model and have the shape of (n, 1)'''

        hidden_act_func = self._hidden_activation_function['act_func']
        hidden_deri = self._hidden_activation_function['deri']
        output_act_func = self._output_activation_function['act_func']
        output_deri = self._output_activation_function['deri']

        sample = sample.reshape(-1, 1)

        a = [self._zero, sample] # sample == a[1]
        grad = [self._zero, self._zero]
        for w in self._weights[2:]:
            z = w @ preprocess.add_bias(a[-1])
            if self._cp:
                g_to_append = cp.array(hidden_deri(z))
                a_to_append = cp.array(hidden_act_func(z))
            else:
                g_to_append = np.array(hidden_deri(z))
                a_to_append = np.array(hidden_act_func(z))
            grad.append(g_to_append)
            a.append(a_to_append)
        a[-1] = output_act_func(z).reshape(-1, 1)
        grad[-1] = output_deri(z).reshape(-1, 1)

        return a, grad

    def train_accuracy(self):
        crr = 0
        for pack in zip(self._x, self._y):
            sample = pack[0] if not self._cp else cp.array(pack[0])
            if self.type == 'classification':
                idx = np.argmax(pack[1]) if not self._cp else np.argmax(pack[1].get())
                y = self._label[idx]  # If the model is classification, y will be a dummies matrix
            else:
                y = pack[1]
            if self.predict(sample, skip_scale=False) == y:
                crr += 1
        return f'{crr / self._NUM_SAMPLES:.2%}'

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
        import time
        self._scale_feature_x()
        for i in range(self.num_epochs):
            print(f'Epoch {i + 1}:')
            start = time.time()
            for idx in range(self._NUM_SAMPLES):
                print(f"{idx+1}/{self._NUM_SAMPLES}", end='\r')
                grad = self._back_prop(self._scaling[0][idx], self._y[idx])
                self._update_weights(grad)
            rand = np.random.randint(0, self._NUM_SAMPLES)
            print(f'''--> Epoch {i + 1} took {(time.time() - start):.2f}s
    Current cost of a random sample: {self._cost_func(sample = self._scaling[0][rand], ground_truth = self._y[rand])[0]}''')
        print(f'Current train accuracy: {self.train_accuracy()}')

    def predict(self, sample, skip_scale = False):
        if not skip_scale:
            sample = self._scale_feature(sample.reshape(-1, self._NUM_NODES[0]))
        res = self._fwd(sample)[0][-1]
        if self._type == 'classification':
            try:
                return self._label[np.argmax(res)]
            except TypeError:
                try:
                    return self._label[np.argmax(res.get())]  # if cupy is used, it's needed to use .get()
                except AttributeError:
                    print(f"Oops something is wrong. res's type is {type(res)}")
                

    # Cost functions; return the value of the cost and its derivative w.r.t its input

    def _mean_square(self, y, ground_truth):
        '''Return: cost, delta'''
        reg = 0
        reg_d = 0
        if self._cp:
            for w in self._weights:
                reg += self._lambda_ * cp.sum(w**2) / 2
                reg_d += self._lambda_ * cp.sum(w) / 2
        else:
            for w in self._weights:
                reg += self._lambda_ * np.sum(w**2) / 2
                reg_d += self._lambda_ * np.sum(w) / 2
        diff = (y - ground_truth).reshape(-1, 1)
        return (diff @ diff.T + reg) / 2, diff

    def _cross_entropy(self, y = None, sample = None, ground_truth = None,):
        '''Return: cost, delta. If both `sample` and 'y' are passed through, `y` will be prioritized'''
        if y is None:
            if sample is not None:
                y = self._fwd(sample)[0][-1]
        reg = 0
        if self._cp:
            for w in self._weights:
                reg += self._lambda_ * cp.sum(w**2) / 2
        else:
            for w in self._weights:
                reg += self._lambda_ * np.sum(w**2) / 2
        def log(val):
            return cp.log(val) if self._cp else np.log(val)
        return -(ground_truth.T @ log(y)) -((1 - ground_truth.T) @ log(1-y)) + reg, y - ground_truth