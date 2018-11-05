class StateManager:
    """ General State Manager class """

    def is_terminal(self):
        """
        Check if state is a terminal state

        :return: True if terminal, else False
        """
        pass

    def generate_child_states(self):
        """
        Generate all child states for given state

        :return: List of child states
        """
        pass


class NimStateManager(StateManager):
    """ Nim State Manager inheriting from the general state manager """

    def __init__(self, p, n, k):
        """
        Initialize Nim state with:

        :param P: Current player
        :param N: Number of pieces/stones left
        :param K: Maximum number of pieces that can be removed
        """
        self.current_player = p  # 1 for Player 1, 2 for Player 2
        self.n = n
        self.k = k

    def is_terminal(self):
        return self.n == 0

    def generate_child_states(self):
        child_states = []
        for pieces in range(1, min([self.k, self.n]) + 1):
            remaining_pieces = self.n - pieces
            child_state = NimStateManager(3-self.current_player, remaining_pieces, self.k)
            child_states.append(child_state)
        return child_states

    def get_n(self):
        """
        Return the amount of remaining pieces

        :return: Number of remaining pieces
        """
        return self.n

    def get_k(self):
        """
        Returns the max amount of pieces that can be removed

        :return: Max amount of pieces that can be removed
        """
        return self.k

    def get_current_player(self):
        """
        Returns the current player

        :return: 1 for Player 1, 2 for Player 2
        """
        return self.current_player

    def get_next_player(self):
        """
        Return next player

        :return: 1 for Player 1, 2 for Player 2
        """
        if self.current_player == 1:
            return 2
        elif self.current_player == 2:
            return 1


