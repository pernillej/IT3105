from gann.gann import Gann
from simulator import Simulator
from gann.replay_buffer import ReplayBuffer
from tournament import TOPP

ACTIVATION_FUNCTION_OPTIONS = ["softmax", "relu", "leaky-relu", "sigmoid", "tanh"]
COST_FUNCTION_OPTIONS = ["mean-squared-error", "cross-entropy"]
OPTIMIZER_OPTIONS = ["gradient-descent", "rmsprop", "adam", "adagrad"]

""" Main interface, specifying user parameters """


def play_best_topp():
    # Hex board
    hex_dim = 5  # 3-8

    # MCTS parameters
    episodes = 300
    starting_player = 1
    rollouts = 100

    # Gann parameters
    learning_rate = 0.001
    gann_dim = [hex_dim ** 2 + 1, 64, 36, hex_dim ** 2]
    hidden_activation_function = "relu"
    cost_function = "mean-squared-error"
    optimizer = "adam"
    minibatch_size = 50

    K = 4  # The number (K) of ANETs to be cached in preparation for a TOPP.
    G = 25  # The number of games, G, to be played between any two ANET-based agents during the round-robin TOPP.

    verbose = False

    # Play TOPP with previously trained actors
    topp = TOPP(games=G, hex_dims=hex_dim, num_actors=K, saved_offset=int(episodes / (K-1)),
                actor_dims=gann_dim, hidden_activation_function=hidden_activation_function,
                cost_function=cost_function,
                learning_rate=learning_rate, optimizer=optimizer, saved_path="bestnetsaver/")
    topp.play_tournament(verbose=verbose)


def main(simulate=True, topp=True):
    # Hex board
    hex_dim = 3  # 3-8

    # MCTS parameters
    episodes = 20
    starting_player = "mix"
    rollouts = 10

    # Gann parameters
    learning_rate = 0.001
    gann_dim = [hex_dim ** 2 + 1, 10, hex_dim ** 2]
    hidden_activation_function = "relu"
    cost_function = "mean-squared-error"
    optimizer = "adam"
    minibatch_size = 50

    K = 5  # The number (K) of ANETs to be cached in preparation for a TOPP.
    G = 10  # The number of games, G, to be played between any two ANET-based agents during the round-robin TOPP.

    verbose = False

    if simulate:
        # Create actor network to run RL
        actor_network = Gann(dimensions=gann_dim, hidden_activation_function=hidden_activation_function,
                             cost_function=cost_function, learning_rate=learning_rate, optimizer=optimizer,
                             minibatch_size=minibatch_size, case=ReplayBuffer())

        # Run reinforcement learning algorithm
        sim = Simulator(actor_network, episodes, starting_player, rollouts, hex_dim,
                        num_saved_actors=K, save_folder="tempnetsaver/")
        sim.simulate(verbose=verbose)

    if topp:  # Play TOPP with just saved network
        topp = TOPP(games=G, hex_dims=hex_dim, num_actors=K, saved_offset=int(episodes / (K-1)),
                    actor_dims=gann_dim, hidden_activation_function=hidden_activation_function,
                    cost_function=cost_function,
                    learning_rate=learning_rate, optimizer=optimizer, saved_path="tempnetsaver/")
        topp.play_tournament(verbose=verbose)


if __name__ == '__main__':
    main(simulate=True, topp=True)
    # play_best_topp()


