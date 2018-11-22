from gann.gann import Gann
from gann.replay_buffer import ReplayBuffer
import tensorflow as tf
import node
from statemanager import HexStateManager


class TOPP:
    """ Tournament of Progressive Policies (TOPP) class """

    def __init__(self, games, hex_dims, num_actors, saved_offset, actor_dims, hidden_activation_function, cost_function,
                 learning_rate, optimizer, saved_path="netsaver/"):
        """
        Initialize Tournament of Progressive Policies (TOPP) object and load actors

        :param games: Number of games in each series
        :param hex_dims: The size of then (n x n) Hexboard , where 3 ≤ n ≤ 8.
        :param num_actors: Number of saved actors to play in tournament
        :param saved_offset: Offset to collect saved network from
        :param actor_dims: Dimensions of actor network layers
        :param hidden_activation_function: Hidden activation function in actor networks
        :param cost_function: Cost function in actor networks
        :param learning_rate: Learning rate in actor networks
        :param optimizer: Optimizer in actor networks
        :param saved_path: Filepath to saved actor networks
        """
        self.games = games
        self.hex_dims = hex_dims
        self.num_actors = num_actors

        self.actor_dims = actor_dims
        self.hidden_activation_function = hidden_activation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.saved_path = saved_path

        self.actors = self.load_actors(saved_offset)

    def load_actors(self, saved_offset):
        """
        Load actors

        :param saved_offset: Offset to collect saved network from
        :return: List of Actor objects representing each saved actor
        """
        actors = []
        for actor in range(self.num_actors):
            actors.append(Actor(self.actor_dims, self.hidden_activation_function, self.cost_function,
                                self.learning_rate, self.optimizer, actor * saved_offset,
                                saved_path=self.saved_path))
        return actors

    @staticmethod
    def rotate_player(current_player):
        """
        Return next player

        :return: 1 for Player 1, 2 for Player 2
        """
        if current_player == 1:
            return 2
        elif current_player == 2:
            return 1

    def play_tournament(self, verbose=False):
        """
        Play The Tournament of Progressive Policies (TOPP)

        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        print("Starting tournament with", len(self.actors), "players")
        print("-" * 49)
        print("-" * 49)
        for i in range(len(self.actors) - 1):
            for j in range(len(self.actors) - 1, i, -1):
                # Get actors for the series
                actor1 = self.actors[i]
                actor2 = self.actors[j]
                starting_player = 1

                # Play assigned number of games
                for game in range(self.games):
                    if verbose:
                        if starting_player == 1:
                            print("Game " + str(game+1) + " - " + str(actor1.name) + " vs. " +
                                  actor2.name)
                        else:
                            print("Game " + str(game+1) + " - " + str(actor2.name) + " vs. " +
                                  actor1.name)

                    # Play single game
                    self.play_game([actor1, actor2], starting_player, verbose=verbose)
                    print("-" * 49)

                    # Rotate starting player
                    starting_player = self.rotate_player(starting_player)

                print("-" * 49)

        print("-" * 49)
        print("FINAL STATISTICS:")
        for actor in self.actors:
            print(actor.name + " won " + str(actor.wins) + "/" + str(self.games*self.num_actors) + " games")
            # Close sessions
            actor.actor.close_current_session()

    def play_game(self, actors, starting_player, verbose):
        """
        Plays single game with two actors

        :param actors: List of 2 Actor objects to play game
        :param starting_player: Starting player
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        # Initialize game state
        root_node = node.Node(parent=None, state=HexStateManager(player=1, dimensions=self.hex_dims))
        root_node.state.generate_initial_state()

        if starting_player == 1:
            players = [actors[0], actors[1]]
        else:
            players = [actors[1], actors[0]]

        current_player = starting_player
        while not root_node.state.is_terminal():
            # Find action for current player
            best_move_node = actors[current_player - 1].get_action(root_node)
            root_node = best_move_node

            # Update player
            current_player = self.rotate_player(current_player)

            if verbose:
                root_node.state.print_state()

        # Get winner
        winner = root_node.state.get_winner()

        # Increase wins
        players[winner - 1].increase_wins()

        if verbose:
            print(str(players[winner - 1].name) + " wins.")


class Actor:
    """ Actor class for Tournament of Progressive Policies (TOPP) play"""

    def __init__(self, dimensions, hidden_ativation_function, cost_function, learning_rate, optimizer, saved_offset,
                 saved_path="netsaver/"):
        """
        Initialize actor object and load saved network

        :param dimensions:
        :param hidden_ativation_function:
        :param cost_function:
        :param learning_rate:
        :param optimizer:
        :param saved_offset:
        :param saved_path:
        """
        self.name = "ANET "
        self.dimensions = dimensions
        self.hidden_activation_function = hidden_ativation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.saved_path = saved_path
        self.saved_offset = saved_offset

        self.wins = 0

        self.load()

    def load(self):
        """ Create network with proper dimensions and settings, and load saved state into network """

        # Initiate network
        self.actor = Gann(dimensions=self.dimensions, hidden_activation_function=self.hidden_activation_function,
                         cost_function=self.cost_function, learning_rate=self.learning_rate, optimizer=self.optimizer,
                         minibatch_size=0, case=ReplayBuffer())

        # Open session
        session = self.actor.open_session()

        # Collect and restore state from saved network
        state_variables = []
        for module in self.actor.modules:
            vars = [module.get_variable('wgt'), module.get_variable('bias')]
            state_variables = state_variables + vars
        self.actor.state_saver = tf.train.Saver(state_variables)
        self.actor.state_saver.restore(sess=session, save_path=(self.saved_path + "-" + str(self.saved_offset)))

        self.name += str(self.saved_offset)

    def get_action(self, node):
        """
        Get the action/child node based on the actors target policy.
        Target Policy: Actor network

        Almost identical to leaf_evaluation function in mcts

        :param node: The node to get an action from
        :return: The proposed best action
        """
        node_indexes = node.state.generate_child_indexes()
        case = node.state.flatten_to_case(node.state.hex_board)  # Not independent from Hex
        actor_prediction = self.actor.predict(case)
        best_move = []
        for i, value in enumerate(actor_prediction):
            if node_indexes[i] == 1:  # If move possible, append
                best_move.append(value)

        # Find the move with the max value
        max_value = max(best_move)
        max_index = best_move.index(max_value)
        children = node.get_children()
        return children[max_index]

    def increase_wins(self):
        """ Increase registered wins by 1 """
        self.wins += 1
