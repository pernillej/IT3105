from gann.gann import Gann
from simulator import Simulator
from gann.replay_buffer import ReplayBuffer

ACTIVATION_FUNCTION_OPTIONS = ["softmax", "relu", "leaky-relu", "sigmoid", "tanh"]
COST_FUNCTION_OPTIONS = ["mean-squared-error", "cross-entropy"]
OPTIMIZER_OPTIONS = ["gradient-descent", "rmsprop", "adam", "adagrad"]

""" Main interface, specifying user parameters """

if __name__ == '__main__':
    # Hex board
    hex_dim = 3  # 3-8

    # MCTS.py parameters
    episodes = 1
    starting_player = 1
    rollouts = 120

    # Gann parameters
    learning_rate = 0.001
    gann_dim = [hex_dim*hex_dim*2 + 2, 64, 32, hex_dim**2]
    hidden_activation_function = "relu"
    cost_function = "mean-squared-error"
    optimizer = "adam"
    minibatch_size = 50

    K = 5  # The number (K) of ANETs to be cached in preparation for a TOPP.
    G = 0  # The number of games, G, to be played between any two ANET-based agents during the round-robin TOPP.

    verbose = True

    # Create actor network to run RL
    actor_network = Gann(dimensions=gann_dim, hidden_activation_function=hidden_activation_function,
                         cost_function=cost_function, learning_rate=learning_rate, optimizer=optimizer,
                         minibatch_size=minibatch_size, case=ReplayBuffer())

    # Run reinforcement learning algorithm
    sim = Simulator(actor_network, episodes, starting_player, rollouts, hex_dim, num_saved_actors=K)
    sim.simulate(verbose=verbose)

    # TODO: Play TOPP

