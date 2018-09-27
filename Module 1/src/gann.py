import tensorflow as tf
import numpy as np
import util.tflowtools as TFT
from gann_module import GannModule
from visualizer import Visualizer


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

        # Map test variables
        self.map_batch_size = map_batch_size
        self.map_layers = map_layers
        self.map_dendrograms = map_dendrograms

        self.modules = []
        self.error_history = []
        self.validation_history = []
        self.grabvars = []

        # Build network
        self.build()

    def add_module(self, module): self.modules.append(module)

    def generate_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """

        self.modules[module_index].gen_probe(type, spec)

    def add_grabvars(self):

        def add_grabvar(grabvars, module_index, grab_type):
            """ Add variable to be displayed by own code"""
            grabvars.append(self.modules[module_index].get_variable(grab_type))

        grabvars = []

        for weight in self.display_weights:
            add_grabvar(grabvars, weight, grab_type='wgt')
        for bias in self.display_biases:
            add_grabvar(grabvars, bias, grab_type='bias')

        return grabvars

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

        # Add grabvars
        # self.grabvars = self.add_grabvars()

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

    def run_one_step(self, operators, grab_vars, session, feed_dict, probed_vars=None):
        """ Run one step in network """

        grab_vars = grab_vars
        if probed_vars is None:
            results = session.run([operators, grab_vars], feed_dict=feed_dict)
        else:
            results = session.run([operators, grab_vars, probed_vars],
                                  feed_dict=feed_dict)
            session.probe_stream.add_summary(results[2])
        return results[0], results[1], session

    def train(self, session, continued=False):
        """ Train network the desired number of steps, with intermittent validation testing """

        if not continued:
            self.error_history = []

        cases = self.case.get_training_cases()
        operators = [self.trainer]
        grab_vars = [self.error] + self.grabvars

        for step in range(1, self.steps + 1):
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

    def get_minibatch(self, cases):
        """ Get minibatch from case-set """

        np.random.shuffle(cases)
        minibatch = cases[:self.minibatch_size]
        return minibatch

    @staticmethod
    def gen_match_counter(logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def test(self, session, cases, msg="Testing", bestk=False):
        """ Test on desired case-set, with our without in-top-k """

        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk:
            # Use in-top-k with k=1
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets], k=1)
        test_res, grabvals, _ = self.run_one_step(self.test_func, [], session=session, feed_dict=feeder)
        if not bestk:
            print('%s Set Error = %f ' % (msg, test_res))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(test_res/len(cases))))
        return test_res  # self.error uses MSE, so this is a per-case value when bestk=None

    def consider_validation_test(self, step, session):
        """ Check to see if validation testing should be done, based on desired validation step """

        if self.validation_interval and (step % self.validation_interval == 0):
            cases = self.case.get_validation_cases()
            if len(cases) > 0:
                error = self.test(session, cases, msg="Validation")
                self.validation_history.append((step, error))

    def test_on_training_set(self, session):
        """ Test on training set """

        cases = self.case.get_training_cases()
        self.test(session, cases, msg="Training", bestk=True)

    @staticmethod
    def open_session(probe=False):
        """ Custom open session """

        if probe:
            return TFT.gen_initialized_session()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess

    @staticmethod
    def close_session(sess, probe=False):
        """ Custom close session """

        if probe:
            TFT.close_session(sess)

        sess.close()

    def run(self):
        """ Run the network and visualize """

        session = self.open_session()
        # Run training and validation testing
        self.train(session)
        # Test on training set
        self.test_on_training_set(session)
        # Test on test set
        self.test(session, self.case.get_testing_cases(), bestk=True)
        # Close session
        self.close_session(session)

        """ Visualization """
        viz = Visualizer()
        viz.plot_error(self.error_history, self.validation_history)
