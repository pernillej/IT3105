from gann import Gann, GannModule
from case import Case
from config import CONFIGS


def main(case):
    """Setup scenario-defining parameters and run

    Keyword arguments:
    case - Config for case to run

    Results:
    1. A plot of the progression of the training-set error from start to end of training. Each data point is the average
    (per case) error for a single mini-batch.
    2. A plot of the progression of the validation-set error from start to end of training.
    3. A listing of the error percentage for the training and test sets, as evaluated after training has finished.

    Additional visualization:
    1. Mapping - This involves taking a small sample of data cases (e.g. 10-20 examples) and running them through the
    network, with learning turned off. The activation levels of a user-chosen set of layers are then displayed for
    each case.
    2. Dendrograms - For any given network layer, a comparison of the different activation vectors (across all cases of
    a mapping) can then serve as the basis for a dendrogram, a convenient graphic indicator of the network’s general
    ability to partition data into relevant groups.
    3. Weight and Bias Viewing - These are simple graphic views of the weights and/or biases associated with the
    connections between any user-chosen pairs of layers. Although you may want to display weights and biases
    intermittently during a run, this assignment only requires their visualization at the end of a run.
    """

    '''
    Setup Scenario Parameters
    '''
    dimensions = case["dimensions"]
    hidden_activation_function = case["hidden_activation_function"]
    output_activation_function = case["output_activation_function"]
    cost_function = case["cost_function"]
    learning_rate = case["learning_rate"]
    init_weight_range = case["init_weight_range"]
    optimizer = case["optimizer"]
    data_source = case["data_source"]
    case_fraction = case["case_fraction"]
    validation_fraction = case["validation_fraction"]
    validation_interval = case["validation_interval"]
    test_fraction = case["test_fraction"]
    minibatch_size = case["minibatch_size"]
    map_batch_size = case["map_batch_size"]
    steps = case["steps"]
    map_layers = case["map_layers"]
    map_dendrograms = case["map_dendrograms"]
    display_weights = case["display_weights"]
    display_biases = case["display_biases"]
    show_interval = case["show_interval"]

    '''
    The Training and Testing Scheme
    '''
    # Create case
    case = Case(data_source, validation_fraction, test_fraction, case_fraction=case_fraction)

    # Build and setup General Artificial Neutral Network
    gann = Gann(dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate,
                init_weight_range, optimizer, case, validation_interval, minibatch_size, steps, display_weights,
                display_biases, map_batch_size=map_batch_size, map_layers=map_layers, map_dendrograms=map_dendrograms,
                show_interval=show_interval)

    # Run training with intermittent validation testing, then test on training set, then test on test set
    gann.run()

    '''
    Visualization
    '''

    # TODO - The sets of corresponding activation levels for user-chosen layers that result from a post-training mapping
    #  run (as described in details file).

    # TODO - Dendrograms (also described details file), which graphically display relationships between input patterns
    # and the hidden-layer activations that they invoke.


CASES = ["custom", "wine", "glass", "yeast", "hackers-choice", "mnist", "parity", "symmetry", "one-hot-autoencoder",
         "dense-autoencoder", "bit-counter", "segment-counter"]


if __name__ == '__main__':

    case = CONFIGS["bit-counter"]

    main(case)



