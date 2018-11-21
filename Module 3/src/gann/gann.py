import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import gann.tflowtools as TFT
from gann.gann_module import GannModule


ACTIVATION_FUNCTIONS = {
    "softmax": tf.nn.softmax,
    "relu": tf.nn.relu,
    "leaky-relu": tf.nn.leaky_relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
}


class Gann:
    """ For setup of entire general artificial neural network and afterwards run cases """

    def __init__(self, dimensions, hidden_activation_function, cost_function, learning_rate, optimizer,
                 minibatch_size, case):

        # Network specifications
        self.dimensions = dimensions
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = "softmax"
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.init_weight_range = [-0.01, 0.1]
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.case = case

        # Internal variables
        self.modules = []

        # Build network
        self.build()

    def add_module(self, module):
        """ Module to list of modules """
        self.modules.append(module)

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
            self.error = tf.losses.mean_squared_error(self.target, self.output)
        elif self.cost_function == "cross-entropy":
            self.error = tf.losses.softmax_cross_entropy(self.target, self.output)
        else:
            raise Exception("Invalid cost function")

        # Setup optimizer
        if self.optimizer == "gradient-descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            raise Exception("Invalid optimizer")

        # Set trainer to minimize error
        self.trainer = optimizer.minimize(self.error, name="Backpropogation")

    def run_one_step(self, operators, feed_dict, grabbed_vars=list(), probed_vars=None, dir='probeview',
                  session=None, step=1):
        """ Run one step in network """

        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:  # When using tensorboard
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        return results[0], results[1], sess

    def train(self, session):
        """ Train network """

        # Get training cases
        cases = self.case.get_cases()

        # Set operator
        operators = [self.trainer]

        # Create minibatch
        minibatch = self.get_minibatch(cases)
        # Turn into feeder dictionary
        inputs = [c[0] for c in minibatch]
        targets = [c[1] for c in minibatch]
        feeder = {self.input: inputs, self.target: targets}

        # Run one step
        _, _, _ = self.run_one_step(operators, feeder, session=session)

    def get_minibatch(self, cases):
        """ Get random minibatch from case-set """
        np.random.shuffle(cases)
        minibatch = cases[:self.minibatch_size]
        return minibatch

    def predict(self, case):
        """
        Run a prediction run to determine the next move

        :param case: The case to run on
        :return: The output array
        """
        r_input = case
        feeder = {self.input: [r_input]}
        return self.current_session.run(self.output, feed_dict=feeder)[0]

    """  UTIL  """

    def open_session(self, probe=False):
        """ Custom open session """

        if probe:
            return TFT.gen_initialized_session()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.current_session = sess
        return sess

    @staticmethod
    def close_session(sess, probe=False):
        """ Custom close session """

        if probe:
            TFT.close_session(sess)

        sess.close()

    def close_current_session(self, probe=False):
        if probe:
            TFT.close_session(self.current_session)
        self.current_session.close()

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.get_variable('wgt'), m.get_variable('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)
