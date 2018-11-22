from node import Node
from statemanager import HexStateManager
from mcts import MCTS
import random


class Simulator:
    """ Simulator class """

    def __init__(self, actor_network, episodes, starting_player, rollouts, hex_dims, num_saved_actors):
        """
        Initiate simulator with specified parameters

        :param actor_network: Neural network to use as actor network
        :param episodes: Number of episodes to run
        :param starting_player: Starting player
        :param rollouts: Number of search games per actual move
        :param hex_dims: The size of then (n x n) Hexboard , where 3 ≤ n ≤ 8.
        :param num_saved_actors: Number of actor to be saved
        """
        self.episodes = episodes
        self.starting_player = starting_player
        self.rollouts = rollouts
        self.hex_dims = hex_dims

        self.actor = actor_network
        self.save_folder = "netsaver/"
        self.save_interval = self.episodes / num_saved_actors  # Save interval for the actor network parameters
        print(self.save_interval)
        self.num_saved_actors = num_saved_actors

    def simulate(self, verbose=False):
        """
        Run reinforcement learning algorithm

        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        mcts = MCTS(self.actor)

        wins_player1 = 0

        # Clear Replay Buffer (RBUF)
        self.actor.case.clear()

        self.actor.open_session()

        # Save actor network state before training
        if self.num_saved_actors != 0:
            self.actor.save_session_params(self.save_folder, self.actor.current_session, 0)
            print("Saved actor on episode 0")

        for i in range(self.episodes):  # For ga in number actual games:
            if self.starting_player == 'mix':
                starting_player = random.randint(1, 2)
            else:
                starting_player = self.starting_player  # 1 for player 1, 2 for player 2

            if verbose:
                print("Game " + str(i) + " - Player " + str(starting_player) + " starting.")

            # Initialize the actual game board (Ba) to an empty board.
            # sinit ← starting board state
            # Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
            root_node = Node(parent=None, state=HexStateManager(player=self.starting_player, dimensions=self.hex_dims))
            root_node.state.generate_initial_state()

            episode_player = starting_player

            # While Ba not in a final state:
            finished = False
            while not finished:
                current_player = root_node.get_state().get_current_player()

                # For gs in number search games:
                # Choose actual move (a*) based on D
                action_node = mcts.get_action(root_node, self.rollouts, episode_player)

                if verbose:
                    action_node.get_state().print_state()

                # D = distribution of visit counts in MCT along all arcs emanating from root.
                visit_counts = []
                for child in root_node.children:
                    visit_counts.append(child.get_visits())
                # Add case (root, D) to RBUF
                indexes = root_node.state.generate_child_indexes()
                hex_case = action_node.state.flatten_to_case(action_node.state.hex_board)
                self.update_replay_buffer(indexes, hex_case, visit_counts)

                # Perform a* on root to produce successor state s*
                # Update Ba to s*
                # In MCT, retain subtree rooted at s*; discard everything else.
                # root←s*
                root_node = action_node

                # Check if game finished
                if root_node.get_state().is_terminal():
                    winner = current_player
                    if verbose:
                        print("Player " + str(winner) + " wins.")
                        print("-" * 49)
                    if winner == 1:
                        wins_player1 += 1
                    finished = True

            # Train ANET on a random minibatch of cases from RBUF
            print("Training network...")
            self.actor.train(self.actor.current_session)
            print("-" * 49)

            # if ga modulo is == 0:
            #   • Save ANET’s current parameters for later use in tournament play.
            if (i + 1) % self.save_interval == 0 and self.num_saved_actors != 0:
                self.actor.save_session_params(self.save_folder, self.actor.current_session, (i + 1))
                print("Saved actor before episode " + str(i+1))

        print("FINAL STATISTICS: ")
        print("Player 1" + " won " + str(wins_player1) + " of " + str(self.episodes) + " games " + " (" + str(
            100 * wins_player1 / self.episodes) + "%)")
        print("Player 2" + " won " + str(self.episodes - wins_player1) + " of " + str(self.episodes) + " games"
              + " (" + str(100 * (self.episodes - wins_player1)/self.episodes) + "%)")
        self.actor.close_current_session(probe=False)

    def update_replay_buffer(self, indexes, hex_case, visit_counts):
        """
        Update the replay buffer

        :param indexes: List of indexes of possible moves
        :param hex_case: Hex state flattened to neural network case
        :param visit_counts: List of visit counts
        :return:
        """
        case = []
        visit_distribution = []
        case.append(hex_case)  # Append the hex board state

        for index in indexes:
            if index == 1:  # Only consider open nodes
                visit_distribution.append(visit_counts.pop(0) - 1)
            else:
                visit_distribution.append(0)

        # Normalize the visit distribution
        normalized_distribution = self.normalize(visit_distribution)

        case.append(normalized_distribution)

        # Push case to the Replay Buffer
        self.actor.case.push(case)

    @staticmethod
    def normalize(distribution):
        """
        Normalize the distribution

        :param distribution: The distribution to normalize
        :return: The normalized distribution
        """
        normalized = []
        one_hot_visit_distribution = [0] * len(distribution)
        # generates one_hot list
        max_value = max(distribution)
        max_index = distribution.index(max_value)
        one_hot_visit_distribution[max_index] = 1

        # generates normalized list
        for value in distribution:
            normalized.append(value / sum(distribution))

        return normalized
