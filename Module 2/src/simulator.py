class Simulator:

    def simulate(self, G, P, M, N, K, verbose=False):
        """
        Simulate one run with specified parameters

        :param G: Number of games in a batch
        :param P: Starting-player option
        :param M: Number of simulations
        :param N: Starting number of pieces/stones in each game
        :param K: Maximum number of pieces that either player can take on their turn
        :param verbose: Verbose mode, if true details of the game are printed (default = False)
        """
        pass


class NimSimulator(Simulator):

    def simulate(self, G, P, M, N, K, verbose=False):
        print("Not implemented")
