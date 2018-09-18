import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import util.tflowtools as TFT


ACTIVATION_FUNCTIONS = {
    "softmax": tf.nn.softmax,
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
    # TODO - Find out if more is relevant?
}

COST_FUNCTIONS = {
    "mean-squared-error": tf.reduce_mean,
    "cross-entropy": tf.reduce_mean
    # TODO - How to do this?
}

OPTIMIZERS = {
    "gradient-descent": tf.train.GradientDescentOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer
    # TODO - Find out if more is relevant? Different input parameters?
}


class Gann:
    """ For setup of entire general artificial neural network and afterwards run cases """

    def __init__(self, dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate,
                 init_weight_range, optimizer, case, validation_interval, minibatch_size, steps, display_weights,
                 display_biases, map_batch_size=0, map_layers=0, map_dendrograms=0):

        self.dimensions = dimensions
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.init_weight_range = init_weight_range
        self.optimizer = optimizer
        self.case = case
        self.validation_interval = validation_interval
        self.minibatch_size = minibatch_size
        self.steps = steps
        self.display_weights = display_weights
        self.display_biases = display_biases

        self.modules = []
        self.global_training_step = 0  # Enables coherent data-storage during extra training runs (see runmore).

        # Map test variables
        self.map_batch_size = map_batch_size
        self.map_layers = map_layers
        self.map_dendrograms = map_dendrograms

        # Build network
        self.build()

    def add_module(self, module): self.modules.append(module)

    def build(self):
        # TODO - build network based on vars
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.dimensions[0]
        # Set input layer of network
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        input_variables = self.input
        input_size = num_inputs
        # Build layer modules
        for i, output_size in enumerate(self.dimensions[1:]):
            g_module = GannModule(self, i, input_variables, input_size, output_size, self.init_weight_range,
                                  self.hidden_activation_function)
            input_variables = g_module.output  # Input for next module is current layers output
            input_size = g_module.output_size
        self.output = g_module.output_size  # Last layer outputs is output of the whole network
        self.output = ACTIVATION_FUNCTIONS[self.output_activation_function](self.output)  # Run activation function
        self.target = tf.placeholder(tf.float64, shape=(None, g_module.output_size), name='Target')
        self.configure_learning()

    def configure_learning(self):
        # TODO - adapt to different cost functions and optimizers
        self.error = COST_FUNCTIONS[self.cost_function](tf.square(self.target - self.output))
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons ????????
        optimizer = OPTIMIZERS[self.optimizer](self.learning_rate)  # Different inputs for other optimizers?
        self.trainer = optimizer.minimize(self.error, name="Backprop")

    def train(self):
        # TODO - run basic training
        return

    def validation_test(self):
        # TODO - run validation test
        return

    def test_on_training_set(self):
        # TODO - after training, test on training set
        return

    def test(self):
        # TODO - test on test set
        return

    def run(self):
        # TODO - function to be called by main.py
        return


class GannModule:
    """ For setup of single gann module = layer of neurons (the output) plus incoming weights and biases """

    def __init__(self, ann, index, input_variables, input_size, output_size, weight_range, activation_function):
        self.ann = ann
        self.index = index
        self.input = input_variables  # Either the gann's input variable or the upstream module's output
        self.input_size = input_size  # Number of neurons feeding into this module
        self.output_size = output_size  # Number of neurons in this module
        self.weight_range = weight_range  # Lower and upper bound for random weight initializing
        self.activation_function = activation_function
        self.name = "Module-" + str(self.index)
        # Build layer
        self.build()

    def build(self):
        name = self.name
        n = self.output_size
        lower_weight_bound = self.weight_range[0]
        upper_weight_bound = self.weight_range[1]
        self.weights = tf.Variable(np.random.uniform(lower_weight_bound, upper_weight_bound, size=(self.input_size, n)),
                                   name=name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=name + '-bias', trainable=True)  # First bias vector
        self.output = ACTIVATION_FUNCTIONS[self.activation_function](tf.matmul(self.input, self.weights) + self.biases,
                                                                     name=name + '-out')
        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def generate_probe(self, type, spec):
        var = self.get_variable(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                    tf.summary.histogram(base + '/hist/', var)

