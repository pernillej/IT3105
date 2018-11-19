import numpy as np


class Case:
    """ For managing cases """

    def __init__(self, cases, validation_fraction, test_fraction, case_fraction=1.0):
        # Case specifications
        self.cases = cases
        self.case_fraction = case_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction + test_fraction)

        self.organize_cases()

    def organize_cases(self):
        """ Organize cases into training, validation and testing, also reduce number of cases if needed """
        cases = self.cases
        np.random.shuffle(cases)  # Randomly shuffle all cases
        if self.case_fraction != 1.0:  # Reduce huge data files
            sep = round(len(self.cases) * self.case_fraction)
            cases = cases[0:sep]

        # Seperate cases
        training_sep = round(len(cases) * self.training_fraction)
        validation_sep = training_sep + round(len(cases) * self.validation_fraction)
        self.training_cases = cases[0:training_sep]
        self.validation_cases = cases[training_sep:validation_sep]
        self.testing_cases = cases[validation_sep:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases

    def get_all_cases(self): return self.training_cases + self.validation_cases + self.testing_cases
