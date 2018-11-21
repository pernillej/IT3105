class ReplayBuffer:
    """ Replay Buffer class """

    def __init__(self, size=500):
        """
        Initialize a replay buffer with designated size

        :param size: Size of replay buffer
        """
        self.size = size
        self.cases = []

    def push(self, case):
        self.cases.append(case)
        if len(self.cases) > self.size:
            self.pop()

    def pop(self):
        return self.cases.pop(0)

    def get_cases(self): return self.cases

    def clear(self):
        self.cases = []
