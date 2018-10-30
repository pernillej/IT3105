"""
TODOs
It should contain code that understands game states and
a) produces initial game states,
b) generates child states from a parent state,
c) recognizes winning states,
and many other functions associated with the domain

"""

from nim import Nim
from nim import NimState
import random


class StateManager:
    """ General State Manager class """

    def __init__(self, game):
        """
        Initialize state manager with a game

        :param game: Game for state manager to handle
        """
        self.game = game

    def is_terminal(self, state):
        """
        Check if state is a terminal state

        :param state: State to check
        :return: True if terminal, else False
        """
        pass

    def generate_child_states(self, state):
        """
        Generate all child states for given state

        :param state: State to generate children from
        :return: List of child states
        """
        pass

    def rollout(self, parent_state, state, M):
        """
        Do a rollout for the given state

        :param parent_state: Parent state of state to perform rollout
        :param state: State to perform rollout
        :param M: Number of rollouts
        :return:
        """
        pass


class NimStateManager(StateManager):
    """ Nim State Manager inheriting from the general state manager """


    def is_terminal(self, state):
        if state.remaining_stones == 0:
            return True

    def generate_child_states(self, state):
        child_states = []  # Change to dict?
        next_player = self.game.get_next_player()

        for pieces in range(1, min([self.game.K, state.remaining_pieces]) + 1):
            remaining_pieces = state.remaining_pieces - pieces
            child_state = NimState(next_player, remaining_pieces)
            child_states.append((child_state, pieces))
        return child_states

    def rollout(self, parent_state, state, M):
        wins = [0, 0]  # Wins for Player 1 at index 0, wins for Player 2 at index 1
        if self.is_terminal(state):
            wins[parent_state.current_player] = M
        else:
            for _ in range(M):
                game = Nim(state.current_player, state.remaining_pieces, self.game.K, verbose=False)

                while not game.finished:  # Behaviour policy here?
                    pieces = random.randint(1, min([self.game.K, game.state.remaining_pieces]))
                    game.select_pieces(pieces)
                wins[game.winner] += 1

        return wins


