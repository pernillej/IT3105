class StateManager:
    """ General State Manager class """

    def __init__(self, starting_player):
        self.current_player = starting_player

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

    def generate_initial_states(self):
        """
        Generate all initial states

        :return: List of initial states
        """
        pass

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


class HexStateManager(StateManager):
    """ Hex Game State Manager """

    def __init__(self, starting_player, dimensions):
        super(HexStateManager, self).__init__(starting_player)

        # Hex specific parameters
        self.dimensions = dimensions

    def is_terminal(self):
        return

    def generate_child_states(self):
        return

    def generate_initial_states(self):
        return

    def print_board(self):
        """
        Print the current state of the Hex board
        """
        return





