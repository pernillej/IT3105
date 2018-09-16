

def main(num_layers, layers_size, hidden_activation_function, output_activation_function, cost_function, learning_rate,
         init_weight_range, optimizer, data_source, case_fraction, validation_fraction, validation_interval,
         test_fraction, minibatch_size, map_batch_size, steps, map_layers, map_dendrograms, display_weights,
         display_biases):
    """Setup scenario-defining parameters and run

    Keyword arguments:
    num_layers --  Number of layers in the network
    layers_size -- List with size corresponding to each layer
    hidden_activation_function -- Function to be used for all hidden layers
    output_activation_function -- Function to be used for output layer
    cost_function -- (a.k.a. loss function) Defines the quantity to be minimized.
    learning_rate -- Learning rate to be used throughout training
    init_weight_range -- Upper and lower bound to be used when randomly initializing all weights (incl. from bias nodes)
    optimizer -- Optimizer to be used for training
    data_source -- Data file or function name to get data files
    case_fraction -- Only use a fraction of overly large data files. (default = 1.0)
    validation_fraction -- The fraction of data cases to be used for validation testing
    validation_interval -- Number of training minibatches between each validation test
    test_fraction -- The fraction of the data cases to be used for standard testing
    minibatch_size -- Number of training cases in a minibatch
    map_batch_size -- Number of training cases to be used for a map test. (Value of 0 means no map tests)
    steps -- Total number of minibatches to be run through the system during training
    map_layers -- The layers to be visualized during the map test, if run
    map_dendrograms -- The layers whose activation patterns (during the map test) will be used to produce dendrograms
    display_weights -- The weight arrays to be visualized at the end of the run
    display_biases -- The bias vectors to be visualized at the end of the run.

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
    The Training and Testing Scheme
    '''

    # TODO - Separate the data into training cases, validation cases and test cases where all cases are assumed to
    # consist of features and labels, i.e., the correct classification.

    # TODO - Repeatedly pass the training features through the ANN to produce an output value, which yields an error
    # term when compared to the correct classification

    # TODO - Use error terms as the basis of backpropagation to modify weights in the network,
    # thus learning to correctly classify the training cases.

    # TODO - Intermittently during training, perform a validation test by turning backpropagation learning off and
    # running the complete set of validation cases through the network one time while recording the average error over
    # those cases.

    # TODO - After a pre-specified number of mini-batches have been processed (or when the total training error has been
    # sufficiently reduced by learning), turn backpropagation off, thereby completing the learning phase.

    # TODO - Run each training case through the ANN one time and record the percentage of correctly-classified cases.

    # TODO - Run each test case through the ANN one time and record the percentage of correctly-classified cases.
    # Use this percentage as an indicator of the trained ANNs ability to generalize to handle new cases
    # (i.e., those that it has not explicitly trained on).

    '''
    Visualization
    '''

    # TODO - A plot showing the progression of training-set and validation-set error (as a function of the training
    # steps, where each step involves the processing of one minibatch).

    # TODO - A display of the weights and biases for any user-chosen areas of the network, for example, the weights
    # between layers 1 and 2 and the biases for layer 2. Typically, these are only shown at the end of the run, but
    # displaying them intermittently is also useful.

    # TODO - The sets of corresponding activation levels for user-chosen layers that result from a post-training mapping
    #  run (as described in details file).

    # TODO - Dendrograms (also described details file), which graphically display relationships between input patterns
    # and the hidden-layer activations that they invoke.


if __name__ == '__main__':
    '''Scenario-defining parameters'''
    num_layers = 0
    layers_size = []
    hidden_activation_function = ""
    output_activation_function = ""
    cost_function = ""
    learning_rate = 0
    init_weight_range = (0, 0)
    optimizer = ""
    data_source = ""
    case_fraction = 1
    validation_fraction = 0
    validation_interval = 0
    test_fraction = 0
    minibatch_size = 0
    map_batch_size = 0
    steps = 0
    map_layers = []
    map_dendrograms = []
    display_weights = []
    display_biases = []

    main(num_layers, layers_size, hidden_activation_function, output_activation_function, cost_function, learning_rate,
         init_weight_range, optimizer, data_source, case_fraction, validation_fraction, validation_interval,
         test_fraction, minibatch_size, map_batch_size, steps, map_layers, map_dendrograms, display_weights,
         display_biases)



