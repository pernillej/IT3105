import random


class Nim:

    def __init__(self, P, N, K, verbose=False):
        """
        Initialize Nim similator with user-specified parameters

        :param P: Starting-player option
        :param N: Starting number of pieces/stones in each game
        :param K: Maximum number of pieces that either player can take on their turn
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        start_player = self.set_start_player(P)  # 0 for Player 1, 1 for Player 2
        self.state = NimState(current_player=start_player, remaining_pieces=N)
        self.K = K
        self.verbose = verbose

        if verbose:
            print("Created new Nim game with N = " + str(self.state.remaining_pieces) + " and K = " + str(self.K) +
                  ". Player " + str(self.state.current_player + 1) + " starts.")

        self.finished = False

    def current_player(self):
        """
        Returns the current player

        :return: 0 for Player 1, 1 for Player 2
        """
        return self.current_player

    def remove_pieces(self, pieces):
        """
        Removes selected amount of pieces from pile

        :param pieces: Number pieces to remove
        """
        if pieces < 1 or pieces > min([self.K, self.state.remaining_pieces]):
            print("Incompatible amount of pieces")
        else:
            self.state.remaining_pieces -= pieces
            if self.verbose:
                print("Player " + str(self.state.current_player + 1) + " selects " + str(pieces) +
                      " pieces: Remaining pieces = " + str(self.state.remaining_pieces))

    def select_pieces(self, pieces):
        """
        Plays one move by a player by removing pieces, checking if finished and changing player at the end

        :param pieces: Number of pieces to remove
        """
        self.remove_pieces(pieces)
        self.check_win()
        self.change_player()

    def check_win(self):
        """
        Checks if game is won and sets appropriate variables

        """
        if self.state.remaining_pieces == 0:
            self.finished = True
            self.winner = self.state.current_player

            if self.verbose:
                print("Player " + str(self.state.current_player + 1) + " won ")

    def change_player(self):
        """
        Change player

        """
        self.state.current_player = (self.state.current_player + 1) % 2

    def get_next_player(self):
        """
        Return next player

        :return: 0 for Player 1, 1 for Player 2
        """
        return (self.state.current_player + 1) % 2

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


class NimState:

    def __init__(self, current_player, remaining_pieces):
        """
        Initialize a nim state

        :param current_player: 0 for Player 1, 1 for Player 2
        :param remaining_pieces: Remaining pieces in pile
        """
        self.current_player = current_player
        self.remaining_pieces = remaining_pieces

    def __eq__(self, other):
        """
        Comparing one state to another

        :param other:
        :return: True if equal, else False
        """
        return self.__dict__ == other.__dict__