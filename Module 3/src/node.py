import random


class Node:

    def __init__(self, parent=None, state=None):
        """
        Initialize a mcts node

        :param parent: Optional parent node
        :param state: Optional state containing information about the state of the game being played
        """
        self.parent = parent
        self.state = state

        self.children = []  # Empty until node expansion is done by mcts
        self.wins = 0
        self.visits = 1  # To avoid zero division

    def get_state(self):
        """
        Get the state associated with this node

        :return: The state associated with this node
        """
        return self.state

    def get_parent(self):
        """
        Get the parent of this node

        :return: The parent of this node, None if root node
        """
        return self.parent

    def get_children(self):
        """
        Get the child nodes from this node

        :return: A list of child nodes
        """
        children = []
        for state in self.state.generate_child_states():
            child = Node(parent=self, state=state)
            children.append(child)
        return children

    def set_children(self, children):
        """
        Set the child nodes of this node. For use during node expansion

        :param children: The children to set as child nodes
        """
        self.children = children

    def get_random_child(self):
        """
        Get a random child from the list of child nodes

        :return: A random child node
        """
        if self.children:
            return random.choice(self.children)
        else:
            return random.choice(self.get_children())

    def get_wins(self):
        """
        Get the amount of wins registered on this node

        :return: The wins registered on this node
        """
        return self.wins

    def increase_wins(self, wins=1):
        """
        Increase the registered wins on this node

        :param wins: The amount of wins to increase with
        """
        self.wins += wins

    def set_wins(self, wins):
        self.wins = wins

    def get_visits(self):
        """
        Get the amount of times this node has been visited

        :return: The amount of times this node has been visited
        """
        return self.visits

    def increase_visits(self, visits=1):
        """
        Increase the registered visits on this node

        :param visits: The amount of visits to increase with
        """
        self.visits += visits

    def set_visits(self, visits):
        self.visits = visits

