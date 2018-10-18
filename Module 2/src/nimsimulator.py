import random

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
        self.G = G
        self.player = self.set_start_player(P)
        self.M = M
        self.N = N
        self.K = K
        self.verbose = verbose

        self.finished = False

        if verbose:
            print("Created new Nim game with N = " + str(self.N) + " and K = " + str(self.K) +
                  ". Player " + str(self.player + 1) + " starts.")

    def remove_pieces(self, pieces):
        """
        Removes selected amount of pieces from pile

        :param pieces: Number pieces to remove
        """
        if pieces < 1 or pieces > self.K or pieces > self.N:
            print("Incompatible amount of pieces")
        else:
            self.N = self.N - pieces
            if self.verbose:
                print("Player " + str(self.player + 1) + " selects " + str(pieces) + " pieces: Remaining pieces = "
                      + str(self.N))

    def select_pieces(self, pieces):
        """
        Plays one move by a player by removing pieces, checking if finished and changing player at the end

        :param pieces: Number of pieces to remove
        """
        self.remove_pieces(pieces)
        self.finished = self.check_win()
        self.change_player()

    def check_win(self):
        """
        Checks if game is won

        :return: True is win, False if not
        """
        if self.N == 0:
            if self.verbose:
                print("Player " + str(self.player + 1) + " won ")
            return True
        else:
            return False

    def is_finished(self):
        return self.finished

    def change_player(self):
        self.player = (self.player + 1) % 2

    @staticmethod
    def set_start_player(P):
        """
        Sets start player based on user input P

        :param P: Option for player to start
        :return: Number of player who starts
        """
        if P == "p1":
            return 0
        elif P == "p2":
            return 1
        else:
            return random.randint(0, 1)

