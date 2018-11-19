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

    def generate_initial_states(self):
        """
        Generate all initial states

        :return: List of initial states
        """
        pass

