import numpy as np
import util.tflowtools as TFT
import data.mnist_basics as mnist

DATA_SET_FUNCTIONS = {
        "parity": lambda: TFT.gen_all_parity_cases(num_bits=10, double=True),
        "symmetry": lambda: TFT.gen_symvect_dataset(vlen=101, count=2000),
        "one-hot-autoencoder": lambda: TFT.gen_all_one_hot_cases(len=8),
        "dense-autoencoder": lambda: TFT.gen_dense_autoencoder_cases(count=2000, size=8, dr=(0.4, 0.7)),
        "bit-counter": lambda: TFT.gen_vector_count_cases(num=500, size=15),
        "segment-counter": lambda: TFT.gen_segmented_vector_cases(count=1000, minsegs=0, maxsegs=8, vectorlen=25)
    }


class Case:
    """ For managing cases """

    def __init__(self, data_source, validation_fraction, test_fraction, case_fraction=1.0):
        self.data_source = data_source
        self.case_fraction = case_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction + test_fraction)

        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        if self.data_source == "mnist":
            self.cases = self.read_from_mnist()
        elif self.data_source == "wine":
            self.cases = self.read_from_file("./data/winequality_red.txt", seperator=";", num_classes=6)
        elif self.data_source == "glass":
            self.cases = self.read_from_file("./data/glass.txt", seperator=",", num_classes=6)
        elif self.data_source == "yeast":
            self.cases = self.read_from_file("./data/yeast.txt", seperator=",", num_classes=10)
        elif self.data_source == "hackers-choice":  # Hackers-choice: Iris data set
            self.cases = self.read_from_file("./data/iris.txt", seperator=",", num_classes=3)
        else:  # For cases to be generated by tflowtools
            self.cases = DATA_SET_FUNCTIONS[self.data_source]()

    def organize_cases(self):
        cases = self.cases
        np.random.shuffle(cases)  # Randomly shuffle all cases
        if self.case_fraction != 1.0:  # Reduce huge data files
            sep = round(len(self.cases) * self.case_fraction)
            cases = cases[0:sep]
        training_sep = round(len(cases) * self.training_fraction)
        validation_sep = training_sep + round(len(cases) * self.validation_fraction)
        self.training_cases = cases[0:training_sep]
        self.validation_cases = cases[training_sep:validation_sep]
        self.testing_cases = cases[validation_sep:]

    def read_from_mnist(self):
        cases = []
        data = mnist.load_all_flat_cases()
        feautures = data[0]
        classes = data[1]

        for i in range(len(feautures)):
            target = [0] * 10
            target[classes[i]] = 1
            cases.append([feautures[i], target])

        return cases

    def read_from_file(self, filename, seperator, num_classes, normalize=True):
        with open(filename, "r") as file:
            raw_data = file.readlines()

            cases = []

            for line in raw_data:

                elements = line.strip("\n").split(seperator)
                target = [0]*num_classes  # Creating empty target vector for case
                classification = elements[-1]  # Getting the case classification, the last element
                case = [float(e) for e in elements[:-1]]

                # If Iris data set, must transform to integer classes
                if self.data_source == "hackers-choice":
                    iris_classes = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
                    target[iris_classes[classification]] = 1
                # If Wine data set, must modify target to start at 0 instead of 3
                elif self.data_source == "wine":
                    target[int(classification) - 3] = 1
                # If Glass set, it doesn't have class 4
                elif self.data_source == "glass":
                    index = int(classification) - 1
                    if index >= 4:  # Must decrease by one
                        target[index - 1] = 1
                    else:
                        target[index] = 1
                else:
                    target[int(classification) - 1] = 1

                # Add case and target to cases array
                cases.append([case, target])

        if not normalize:
            return cases

        # Else normalize case features
        return self.normalize_cases(cases)

    def normalize_cases(self, cases):
        maximum = [0] * len(cases[0][0])
        minimum = [np.inf] * len(cases[0][0])
        for c in cases:
            index = 0
            for feature in c[0]:
                if feature > maximum[index]:
                    maximum[index] = feature
                elif feature < minimum[index]:
                    minimum[index] = feature
                index += 1

        for c in range(len(cases)):
            for j in range(len(cases[c][0])):
                cases[c][0][j] = (cases[c][0][j] - minimum[j]) / (maximum[j] - minimum[j])

        return cases

    ''' Getters '''

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases

    def get_all_cases(self): return self.training_cases + self.validation_cases + self.testing_cases
