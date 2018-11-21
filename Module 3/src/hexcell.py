class HexCell:

    def __init__(self):
        self.value = [0, 0]
        self.position = []

    def set_position(self, position):
        self.position = position

    def is_empty(self):
        return self.value == [0, 0]

    def get_player(self):
        if self.value == [0, 0]:
            return 0
        elif self.value == [1, 0]:
            return 1
        else:
            return 2