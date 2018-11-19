ACTIVATION_FUNCTION_OPTIONS = ["softmax", "relu", "leaky-relu", "sigmoid", "tanh"]
COST_FUNCTION_OPTIONS = ["mean-squared-error", "cross-entropy"]
OPTIMIZER_OPTIONS = ["gradient-descent", "rmsprop", "adam", "adagrad"]

""" Main interface, specifying user parameters """

if __name__ == '__main__':
    # Hex board
    hex_dim = 5  # 3-8

    # MCTS parameters
    episodes = 10
    starting_player = 1
    rollouts = 1000

    # Gann parameters
    learning_rate = 0
    gann_dim = []
    hidden_activation_function = ""
    output_activation_function = ""
    cost_function = ""
    optimizer = ""

    K = 0  # The number (K) of ANETs to be cached in preparation for a TOPP.
    G = 0  # The number of games, G, to be played between any two ANET-based agents during the round-robin TOPP.

    verbose = False

    # TODO: Run RL

    # TODO: Play TOPP
