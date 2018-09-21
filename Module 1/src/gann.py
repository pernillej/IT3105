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
        self.error_history = []
        self.validation_history = []

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
        # TODO - Customize more?
        if self.optimizer == "gradient-descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            '''
            learning_rate: A Tensor or a floating point value. The learning rate to use.
            use_locking: If True use locks for update operations.
            '''
        elif self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
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

    def run_one_step(self, operators, grab_vars, session, feed_dict, probed_vars=None):
        grab_vars = grab_vars + self.display_weights + self.display_biases
        if probed_vars is None:
            results = session.run([operators, grab_vars], feed_dict=feed_dict)
        else:
            results = session.run([operators, grab_vars, probed_vars],
                                  feed_dict=feed_dict)
            session.probe_stream.add_summary(results[2])
        return results[0], results[1], session

    '''
     In project 1, there is no need to go systematically through the training data.  
     If the user specifies 1000 "steps" of training, that means 1000 minibatches (of size M, with M also specified by 
     the user), each drawn randomly from the training data.  There is no need for epochs in project 1.  You are free to 
     use them if you want, but it's not a requirement.
    '''
    def train(self, session, continued=False):
        if not continued:
            self.error_history = []

        cases = self.case.get_training_cases()
        operators = [self.trainer]
        grab_vars = [self.error]

        for step in range(self.steps):
            # Create minibatch
            minibatch = self.get_minibatch(cases)
            # Turn into feeder dictionary
            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}

            # Run one step
            _,grabvals,_ = self.run_one_step(operators, grab_vars, session, feeder)

            # Append error to error history
            self.error_history.append((step, grabvals[0]))

            # Consider validation testing
            self.consider_validation_test(step, session)

        TFT.plot_training_history(self.error_history, self.validation_history, xtitle="Epoch", ytitle="Error",
                                  title="", fig=not continued)

    def get_minibatch(self, cases):
        num_cases = len(cases)
        # Get random number between 0 and the highest start index possible, to get desired minibatch size
        mbstart = np.random.randint(0, num_cases-self.minibatch_size)
        minibatch = cases[mbstart:mbstart+self.minibatch_size]
        return minibatch

    def test(self, session, cases):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        test_res, grabvals, _ = self.run_one_step(self.error, [], session, feeder)
        print('%s Set Correct Classifications = %f %%' % ("Testing", 100 * (test_res / len(cases))))
        return test_res

    def consider_validation_test(self, step, session):
        if self.validation_interval and (step % self.validation_interval == 0):
            cases = self.case.get_validation_cases()
            if len(cases) > 0:
                error = self.test(session, cases)
                self.validation_history.append((step, error))

    def test_on_training_set(self):
        # TODO - after training, test on training set
        return

    def run(self):
        session = TFT.gen_initialized_session(dir='probeview')
        # Run training and validation testing
        self.train(session)
        # Test on training set
        self.test_on_training_set()
        # Test on test set
        self.test(session, self.case.get_training_cases())
        # Close session
        TFT.close_session(session)
