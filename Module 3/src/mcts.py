import numpy as np
import random


class MCTS:
    """ Monte Carlo Tree Search class """

    def __init__(self, gann):
        self.gann = gann

    def get_action(self, node, M, starting_player):
        """
        Return the action/child of the given node with the best Q(s,a) score

        :param node: The node to get an action from
        :param M: The amount of rollouts/simulations per episode
        :param starting_player: The starting player of the batch of games
        :return: The proposed best action, the amount of pieces the action would remove,
        and the remaining pieces after the action is performed
        """
        self.update(node, M, starting_player)  # Update the tree using mcts

        current_player = node.get_state().get_current_player()
        action_node = None

        highest_qsa = -float('inf')
        lowest_qsa = float('inf')

        for child in node.children:
            qsa = float(child.get_wins())/float(child.get_visits())  # Calculate Q(s,a)

            if current_player == starting_player:
                if qsa > highest_qsa:
                    highest_qsa = qsa
                    action_node = child
            else:  # If the current player is the opposing player, the best score is the lowest Q(s,a)
                if qsa < lowest_qsa:
                    lowest_qsa = qsa
                    action_node = child

        return action_node

    def update(self, node, M, current_player):
        """
        Update the internal state of the mcts tree, by running mcts on a given node

        :param node: The root node to run mcts on
        :param M: The amount of rollouts/simulations per episode
        :param current_player: The current player
        """
        for i in range(0, M):

            # Use tree policy to chose the next pre-existing child node.
            best_node = self.tree_search(node, current_player)

            # When a leaf node (not necessarily a final state) is reached,
            # apply domain-dependent operators to produce successor (child) states
            self.node_expansion(best_node)
            if len(best_node.get_children()) > 0:  # Chose a random child if just expanded
                best_node = random.choice(best_node.get_children())

            # Use the default policy to perform a single-track rollout simulation from S to a final state (F).
            winner = self.leaf_evaluation(best_node)

            # Propagate information about F (such as the winner), along the entire path from F back to S and then
            # back to R. That info updates node statistics that influence the tree policy.
            self.backpropagate(best_node, winner, current_player)

    @staticmethod
    def tree_policy_value(parent, child, opposing_player):
        """
        Return the value of the action based on the tree policy.
        Adds Q(s,a) value, i.e. the value (expected final result) of choosing action a from state s,
        with u(s,a) value, which is the exploration bonus.
        u(s,a) is based on the UCT metric.

        :param parent: The node representing the current state
        :param child: The node representing a specific action taken from the current state
        :param opposing_player: True if the player is the opposing player, which means the player will chose low values
        :return: Return the value of the action resulting in moving from node to child
        """
        qsa = child.get_wins() / child.get_visits()
        usa = 1 * np.sqrt(np.log(parent.get_visits()) / (1 + child.get_visits()))

        if opposing_player:
            usa *= -1

        return qsa + usa

    def tree_search(self, node, current_player):
        """
        Traverse the tree from the given node, based on the tree policy

        :param node: Node to traverse from
        :param current_player: The current player
        :return: Returns a leaf node based on the tree policy
        """
        if not node.children:  # Breaks recursion and returns the best leaf node
            return node

        best_child = node
        highest_value = -float('inf')
        lowest_value = float('inf')
        opposing_player = node.get_state().get_current_player() != current_player

        for child in node.children:
            value = self.tree_policy_value(node, child, opposing_player)  # Get value of node based on the tree policy

            if opposing_player and value < lowest_value:
                # The best value is the lowest value when the player is the opposing player
                best_child = child
                lowest_value = value

            elif (not opposing_player) and value > highest_value:
                best_child = child
                highest_value = value
        return self.tree_search(best_child, current_player)  # Recursively search the tree until reaching best leaf node

    @staticmethod
    def node_expansion(node):
        """
        Expands a node by generating it's children

        :param node: Node to do expansion from
        """
        if not node.children:
            node.set_children(node.get_children())

    def leaf_evaluation(self, node):
        """
        Evaluate the leaf node based on the result of a rollout using the default/behavior policy.
        Default policy: Actor network

        :param node: Leaf node to evaluate
        :return: Returns the winner
        """
        while not node.state.is_terminal():
            node_indexes = node.state.generate_child_indexes()
            case = node.state.flatten_to_case(node.state.hex_board)  # Not independent from Hex
            actor_prediction = self.gann.predict(case)
            best_move = []
            for i, value in enumerate(actor_prediction):
                if node_indexes[i] == 1:  # If move possible, append
                    best_move.append(value)

            # Find the move with the max value
            max_value = max(best_move)
            max_index = best_move.index(max_value)
            children = node.get_children()
            node = children[max_index]

        winner = node.get_state().get_winner()
        return winner

    @staticmethod
    def backpropagate(node, winner, current_player):
        """
        Backpropagate information about the node along the path of the node back to the root

        :param node: The node that was evaluated
        :param winner: The winner of the evaluation rollout of said node
        :param current_player: The current player
        """
        while node is not None:
            if winner == current_player:
                node.increase_wins(wins=1)
            node.increase_visits(visits=1)
            node = node.parent
