from node import Node
from mcts import MCTS
from statemanager import NimStateManager
import random


class Simulator:

    def simulate(self, G, P, M, N, K, verbose=False):
        """
        Simulate one run with specified parameters

        :param G: Number of games in a batch
        :param P: Starting-player option
        :param M: Number of simulations
        :param N: Starting number of pieces/stones in each game
        :param K: Maximum number of pieces that either player can take on their turn
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        pass


class NimSimulator(Simulator):
    """ Nim simulator inheriting from general simulator"""

    def simulate(self, G, P, M, N, K, verbose=False):
        mcts = MCTS()

        wins_player1 = 0

        for i in range(0, G):
            if P == 'mix':
                starting_player = random.randint(1, 2)
                print(starting_player)
            else:
                starting_player = P  # 1 for player 1, 2 for player 2

            root_node = Node(parent=None, state=NimStateManager(p=starting_player, n=N, k=K))

            game_over = False

            while not game_over:

                current_player = root_node.get_state().get_current_player()
                next_player = root_node.get_state().get_next_player()
                action_node, removed_pieces, remaining_pieces = mcts.get_action(root_node, M, starting_player)

                if verbose:
                    print("Player " + str(current_player) + " selected " + str(removed_pieces) + " pieces."
                          + " Remaining pieces = " + str(remaining_pieces))

                root_node = Node(state=NimStateManager(p=next_player, n=remaining_pieces, k=K))

                if root_node.get_state().is_terminal():
                    winner = current_player  # The winner is the previous player
                    if verbose:
                        print("Player " + str(winner) + " wins.")
                        print("-"*49)
                    if winner == 1:
                        wins_player1 += 1
                    game_over = True

        print("FINAL STATISTICS: ")
        print("Player 1" + " wins " + str(wins_player1) + " of " + str(G) + " games" + " (" + str(
            100 * wins_player1 / G) + "%).")
        print("Player 2" + " wins " + str(G - wins_player1) + " of " + str(G) + " games" + " (" + str(
            100 * (G - wins_player1) / G) + "%).")

