from simulator import NimSimulator

""" Main interface, specifying user parameters """

if __name__ == '__main__':
    # User-specified parameters
    G = 10
    P = 1
    M = 1000
    N = 10
    K = 3
    verbose = False

    # Setup simulator
    sim = NimSimulator()

    # Simulate one run of user parameters
    sim.simulate(G=G, P=P, M=M, N=N, K=K, verbose=verbose)

