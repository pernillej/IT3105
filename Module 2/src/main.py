from Simulator import NimSimulator

""" Main interface, specifying user parameters """

if __name__ == '__main__':
    # User-specified parameters
    G = 0
    P = 0
    M = 0
    N = 0
    K = 0
    verbose = False

    # Setup simulator
    sim = NimSimulator()

    # Simulate one run of user parameters
    sim.simulate(G=G, P=P, M=M, N=N, K=K, verbose=verbose)
