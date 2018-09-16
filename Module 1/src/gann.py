import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import util.tflowtools as TFT


class Gann:
    """ For setup of entire general artificial neural network and afterwards run cases """

    def __init__(self):
        # TODO - initialize vars
        return

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

    def __init__(self, cfunc, vfrac=0, tfrac=0):
        self.case_function = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)

        self.cases = None
        self.training_cases = None
        self.validation_cases = None
        self.testing_cases = None
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases

