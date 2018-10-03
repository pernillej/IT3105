import tensorflow as tf
import numpy as np
import util.tflowtools as TFT


class GannModule:
    """ For setup of single gann module = layer of neurons (the output) plus incoming weights and biases """

    def __init__(self, ann, index, input_variables, input_size, output_size, weight_range, activation_function):
        self.ann = ann
        self.index = index
        self.input = input_variables  # Both the gann's input variable or the upstream module's output
        self.input_size = input_size  # Number of neurons feeding into this module
        self.output_size = output_size  # Number of neurons in this module
        self.weight_range = weight_range  # Lower and upper bound for random weight initializing
        self.activation_function = activation_function
        self.name = "Module-" + str(self.index)

        # Build module
        self.build()

    def build(self):
        name = self.name
        lower_weight_bound = self.weight_range[0]
        upper_weight_bound = self.weight_range[1]

        # Setup weights
        self.weights = tf.Variable(np.random.uniform(lower_weight_bound, upper_weight_bound,
                                                     size=(self.input_size, self.output_size)),
                                   name=name + '-wgt', trainable=True)  # True = default for trainable anyway

        # Setup biases
        self.biases = tf.Variable(np.random.uniform(lower_weight_bound, upper_weight_bound,
                                                    size=(1, self.output_size)),
                                  name=name + '-bias', trainable=True)

        # Setup outputs
        self.output = self.activation_function(tf.matmul(self.input, self.weights) + self.biases, name=name + '-out')

        # Add to list of modules in ann
        self.ann.add_module(self)

    def get_variable(self, type):  # type = (in,out,wgt,bias)
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

