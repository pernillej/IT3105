import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import util.tflowtools as TFT
from gann_module import GannModule


ACTIVATION_FUNCTIONS = {
    "softmax": tf.nn.softmax,
    "relu": tf.nn.relu,
    "leaky-relu": tf.nn.leaky_relu,
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
        self.show_interval = show_interval  # Frequency of showing grabbed variables

        # Build network
        self.build()

    def add_module(self, module): self.modules.append(module)

    def generate_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """

        self.modules[module_index].gen_probe(type, spec)

    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].get_variable(type))

    def add_grabvars(self):
        for weight in self.display_weights:
            self.add_grabvar(weight, type='wgt')
        for bias in self.display_biases:
            self.add_grabvar(bias, type='bias')

        # self.add_grabvar(2, type='out')

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
            self.display_grabvars(step=step, feed_dict=feed_dict)
            self.print_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def print_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", v, end="\n")

    def display_grabvars(self, step=1, feed_dict=None):
        w_and_b = []
        for w in self.display_weights:
            w_and_b.append(self.modules[w].get_variable('wgt'))
        for b in self.display_weights:
            w_and_b.append(self.modules[b].get_variable('bias'))
        # w_and_b.append(self.modules[1].get_variable('out'))
        if len(w_and_b) != 0:
            print("Creating figures for grabbed variables at step " + str(step))
            names = [x.name for x in w_and_b]
            for i, v in enumerate(w_and_b):
                matrix = self.current_session.run(v, feed_dict=feed_dict)
                if type(matrix) == np.ndarray and len(v.shape) > 1:
                    title = names[i] + " step: " + str(step)
                    TFT.display_matrix(matrix, title=title)
                else:
                    print(v, end="\n\n")

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

        if (self.validation_interval and (step % self.validation_interval == 0)) or step == 1:
            cases = self.case.get_validation_cases()
            if len(cases) > 0:
                error = self.test(session, cases, msg="Validation")
                self.validation_history.append((step, error))

    def test_on_training_set(self, session):
        """ Test on training set """

        cases = self.case.get_training_cases()
        self.test(session, cases, msg="Training", bestk=True)

    def mapping(self):
        """ Mapping test """

        cases = self.case.get_all_cases()
        np.random.shuffle(cases)
        batch = cases[:self.map_batch_size]

        grabvals = []

        for case in batch:
            inputs = [case[0]]
            targets = [case[1]]
            feeder = {self.input: inputs, self.target: targets}

            grabvars = []
            layers = len(self.dimensions) - 1
            for layer in range(layers):
                grabvars.append(self.modules[layer].get_variable('out'))
            results = self.current_session.run([self.predictor, grabvars], feed_dict=feeder)

            grabvals_per_case = []
            for i, v in enumerate(results[1]):
                grabvals_per_case.append(v[0])

            grabvals.append(grabvals_per_case)

        grabvals_per_layer = []
        for layer_index in range(len(self.dimensions) - 1):
            grabvals_per_layer.append([])
            for i in range(len(batch)):
                grabvals_per_layer[layer_index].append(grabvals[i][layer_index])

        # Hinton plots
        if len(self.map_layers) != 0:
            TFT.hinton_plot(np.array([c[0] for c in batch]), title="Hinton plot inputs")
            TFT.hinton_plot(np.array([c[1] for c in batch]), title="Hinton plot targets")
        for layer in self.map_layers:
            print("Creating hinton figure for layer " + str(layer))
            TFT.hinton_plot(np.array(grabvals_per_layer[layer]), title="Hinton plot layer" + str(layer))

        # Dendograms
        labels = [TFT.bits_to_str(c[0]) for c in batch]
        for layer in self.map_dendrograms:
            print("Creating dendrogram figure for layer " + str(layer))
            PLT.figure()
            TFT.dendrogram(grabvals_per_layer[layer], labels, title="Dendrogram with inputs of layer " + str(layer))

    def plot_error_and_validation_history(self):
        fig = PLT.figure()
        fig.suptitle('Error and Validation', fontsize=18)
        # Change to proper format
        e_xs = [e[0] for e in self.error_history]
        e_ys = [e[1] for e in self.error_history]
        v_xs = [e[0] for e in self.validation_history]
        v_ys = [e[1] for e in self.validation_history]
        PLT.plot(e_xs, e_ys, label='Error history')
        PLT.plot(v_xs, v_ys, label='Validation error history')
        PLT.xlabel("Steps")
        PLT.ylabel("Error")
        PLT.legend()


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
        self.plot_error_and_validation_history()

        """ Mapping test """
        if len(self.map_layers) != 0 or len(self.map_dendrograms) != 0:
            print("\n", "**** Running mapping test ***** ", "\n")
            self.mapping()

        PLT.show()

        # Close session
        self.close_session(self.current_session)
