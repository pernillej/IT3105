import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import util.tflowtools as TFT


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

    def build(self):
        # TODO - build network based on vars
        return

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


class GannLayer:
    """ For setup of single gann layer """

    def __init__(self):
        # TODO - initialize layer vars
        return

    def build(self):
        # TODO - build layer based on vars
        return


class Case:
    """ For managing cases """

    def __init__(self, data_source, validation_fraction, test_fraction, case_fraction=1.0):
        self.data_source = data_source
        self.case_fraction = case_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction + test_fraction)

        self.cases = None
        self.training_cases = None
        self.validation_cases = None
        self.testing_cases = None
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.data_source()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        cases = np.array(self.cases)
        np.random.shuffle(cases)  # Randomly shuffle all cases
        if self.case_fraction != 1.0:  # Reduce huge data files
            sep = round(len(self.cases) * self.case_fraction)
            cases = cases[0:sep]
        training_sep = round(len(cases) * self.training_fraction)
        validation_sep = training_sep + round(len(cases) * self.validation_fraction)
        self.training_cases = cases[0:training_sep]
        self.validation_cases = cases[training_sep:validation_sep]
        self.testing_cases = cases[validation_sep:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases

    # TODO - implement data_source functions
