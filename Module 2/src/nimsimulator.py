from enum import Enum

"""
Given N and K, the remaining rules of play are extremely simple: Players take turns removing pieces, 
and the player who removes the last piece is the winner.

TODOs
Simulate single game
Verbose mode
Batch runs


The results that it must provide:
• Play-by-Play game action when in verbose mode.
• Essential win statistics (for at least one of the two players) for a batch of games.
"""


class NimSimulator:

    def __init__(self, G, P, M, N, K, verbose=False):
        """
        Initialize Nim similator with user-specified parameters

        :param G: Number of games in a batch
        :param P: Starting-player option
        :param M: Number of simulations (and hence rollouts) per actual game move
        :param N: Starting number of pieces/stones in each game
        :param K: Maximum number of pieces that either player can take on their turn
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        return


class POptions(Enum):
    """
    Enum for starting player options
    """

    Player1 = 0
    Player2 = 0
    Mix = 3
