import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import util.tflowtools as TFT
from gann_module import GannModule


ACTIVATION_FUNCTIONS = {
    "softmax": tf.nn.softmax,
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
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
        """ Build network from input layer to output layer with all hidden layers """

        tf.reset_default_graph()  # This is essential for doing multiple runs!!

        # Set input layer of network
        num_inputs = self.dimensions[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')

        # Build layer modules
        input_variables = self.input
        input_size = num_inputs
        output_layer = len(self.dimensions) - 1  # To be used when building the final layer

        for i, output_size in enumerate(self.dimensions[1:]):
            # If building output layer(aka final layer) set activation function to output_activation_function
            if i == output_layer:
                activation_function = ACTIVATION_FUNCTIONS[self.output_activation_function]
            # Else use hidden layer activation function
            else:
                activation_function = ACTIVATION_FUNCTIONS[self.hidden_activation_function]

            # Build module
            g_module = GannModule(self, i, input_variables, input_size, output_size, self.init_weight_range,
                                  activation_function)

            # Set input variables for next layer
            input_variables = g_module.output
            input_size = g_module.output_size

        # Last layer outputs is output of the whole network
        self.output = g_module.output

        # Setup target vector
        self.target = tf.placeholder(tf.float64, shape=(None, g_module.output_size), name='Target')

        # Configure learning
        self.configure_learning()

    def configure_learning(self):
        """ Configure learning for network """

        self.predictor = self.output  # Simple prediction runs will request the value of output neurons

        # Setup error function
        if self.cost_function == "mean-squared-error":
            self.error = tf.reduce_mean(tf.square(self.target - self.output),
                                        name="Mean-squared-error")
        elif self.cost_function == "cross-entropy":
            self.error = - tf.reduce_mean(tf.reduce_sum(self.target * tf.log(self.output), [1]),
                                          name='Cross-entropy-error')
        else:
            raise Exception("Invalid cost function")

        # Setup optimizer
        if self.optimizer == "gradient-descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            '''
            learning_rate: A Tensor or a floating point value. The learning rate to use.
            use_locking: If True use locks for update operations.
            '''
        elif self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            # TODO - Customize more?
            '''
            learning_rate: A Tensor or a floating point value. The learning rate.
            decay: Discounting factor for the history/coming gradient
            momentum: A scalar tensor.
            epsilon: Small value to avoid zero denominator.
            use_locking: If True use locks for update operation.
            centered: If True, gradients are normalized by the estimated variance of the gradient; if False, by the 
            uncentered second moment. Setting this to True may help with training, but is slightly more expensive in 
            terms of computation and memory. Defaults to False.
            '''
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # TODO - Customize more?
            '''
            learning_rate: A Tensor or a floating point value. The learning rate.
            beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper 
            (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
            use_locking: If True use locks for update operations.
            '''
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
            # TODO - Customize more?
            '''
            learning_rate: A Tensor or a floating point value. The learning rate.
            initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
            use_locking: If True use locks for update operations.
            '''
        else:
            raise Exception("Invalid optimizer")

        # Set trainer to minimize error
        self.trainer = optimizer.minimize(self.error, name="Backpropogation")

    def generate_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """
        self.modules[module_index].gen_probe(type, spec)

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
