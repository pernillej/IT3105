from Simulator import NimSimulator

""" Main interface, specifying user parameters """

if __name__ == '__main__':
    user_parameters = {
        "g": 0,
        "p": 0,
        "m": 0,
        "n": 0,
        "k": 0,
        "verbose": False
    }

    sim = NimSimulator()
    sim.simulate_batch(**user_parameters)
