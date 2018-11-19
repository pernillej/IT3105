from node import Node
from mcts import MCTS
from statemanager import NimStateManager
import random


class Simulator:

    def simulate(self, episodes, starting_player, rollouts, hex_dims, verbose=False):
        """
        Simulate one run with specified parameters, using reinforcement learning

        :param episodes: Number of episodes to run
        :param starting_player: Starting player
        :param rollouts: Number of search games per actual move
        :param hex_dims: The size of then (n x n) Hexboard , where 3 ≤ n ≤ 8.
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        pass


