import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import util.tflowtools as TFT
from .gann_module import GannModule


ACTIVATION_FUNCTIONS = {
    "softmax": tf.nn.softmax,
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
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
