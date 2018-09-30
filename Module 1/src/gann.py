import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
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
                 display_biases, map_batch_size=0, map_layers=None, map_dendrograms=None, show_interval=1):

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
        self.grabvar_figures = []  # One matplotlib figure for each grabvar
        self.show_interval = show_interval  # Frequency of showing grabbed variables

        # Build network
        self.build()

    def add_module(self, module): self.modules.append(module)

    def generate_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """

        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].get_variable(type))
        self.grabvar_figures.append(PLT.figure())

    def add_grabvars(self):
        for weight in self.display_weights:
            self.add_grabvar(weight, type='wgt')
        for bias in self.display_biases:
            self.add_grabvar(bias, type='bias')

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

        # Setup grabvars
        self.add_grabvars()

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
            self.error = - tf.reduce_mean(tf.losses.softmax_cross_entropy(self.target, self.predictor),
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

    def run_one_step(self, operators, feed_dict, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, step=1, show_interval=1):
        """ Run one step in network """

        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", v, end="\n")

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
            _,grabvals,_ = self.run_one_step(operators, feeder, grab_vars, session=session, step=step,
                                             show_interval=self.show_interval)

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
        test_res, grabvals, _ = self.run_one_step(self.test_func, feeder, [], session=session, show_interval=None)
        if not bestk:
            print('%s Set Error = %f ' % (msg, test_res))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(test_res/len(cases))))
        return test_res

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

    def mapping(self):
        self.add_mapping_grabvars()
        names = [x.name for x in self.grabvars]
        cases = self.case.get_testing_cases()[:self.map_batch_size]
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        results = self.current_session.run([self.output, self.grabvars], feed_dict=feeder)
        fig_index = 0
        for i, v in enumerate(results[1]):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1:  # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v, fig=self.grabvar_figures[fig_index], title=names[i] + "mapping")
                fig_index += 1
            else:
                print(v, end="\n\n")

    def add_mapping_grabvars(self):
        self.grabvars = []
        for layer in self.map_layers:
            self.add_grabvar(layer, type='wgt')


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

        self.current_session = self.open_session()
        # Run training and validation testing
        self.train(self.current_session)
        # Test on training set
        self.test_on_training_set(self.current_session)
        # Test on test set
        self.test(self.current_session, self.case.get_testing_cases(), bestk=True)

        """ Visualization """
        viz = Visualizer()
        viz.plot_error(self.error_history, self.validation_history)

        """ Mapping test """
        if self.map_layers is not None:
            self.mapping()
            PLT.show()

        # Close session
        self.close_session(self.current_session)
