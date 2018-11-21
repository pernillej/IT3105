from hexcell import HexCell
import pickle
import numpy as np


class StateManager:
    """ General State Manager class """

    def __init__(self, player):
        self.player = player

    def generate_initial_state(self):
        """
        Generate all initial states

        :return: List of initial states
        """
        pass

    def generate_child_states(self):
        """
        Generate all child states for given state

        :return: List of child states
        """
        pass

    def is_terminal(self):
        """
        Check if state is a terminal state

        :return: True if terminal, else False
        """
        pass

    def get_current_player(self):
        """
        Returns the current player

        :return: 1 for Player 1, 2 for Player 2
        """
        return self.player

    def get_next_player(self):
        """
        Return next player

        :return: 1 for Player 1, 2 for Player 2
        """
        if self.player == 1:
            return 2
        elif self.player == 2:
            return 1


class HexStateManager(StateManager):
    """ Hex Game State Manager """

    def __init__(self, player, dimensions, hex_board=None, neighbours=None):
        super(HexStateManager, self).__init__(player=player)
        self.winner = None

        # Parameters relating to Hex
        self.hex_board = hex_board
        self.dimensions = dimensions
        self.neighbours = neighbours

    def generate_initial_state(self):
        dimensions = self.dimensions
        hex_board = []
        for row in range(dimensions):
            row_list = []
            for col in range(dimensions):
                row_list.append(HexCell())
            hex_board.append(row_list)

        self.hex_board = hex_board
        self.neighbours = self.generate_neighbours(hex_board)

    def generate_neighbours(self, hex_board):
        neighbours_dict = {}

        board_length = len(hex_board) - 1
        for i, row in enumerate(self.hex_board):
            for j, cell in enumerate(row):
                position = [i, j]
                neighbours = []
                if 0 <= i - 1 <= board_length:
                    neighbours.append([i - 1, j])
                if 0 <= i - 1 <= board_length and 0 <= j + 1 <= board_length:
                    neighbours.append([i - 1, j + 1])
                if 0 <= j - 1 <= board_length:
                    neighbours.append([i, j - 1])
                if 0 <= j + 1 <= board_length:
                    neighbours.append([i, j + 1])
                if 0 <= i + 1 <= board_length and 0 <= j - 1 <= board_length:
                    neighbours.append([i + 1, j - 1])
                if 0 <= i + 1 <= board_length:
                    neighbours.append([i + 1, j])
                neighbours_dict[str(position)] = neighbours
                cell.set_position([i, j])
        return neighbours_dict

    def generate_child_states(self):
        children = []

        for i, row in enumerate(self.hex_board):
            for j, cell in enumerate(row):
                copy_board = pickle.loads(pickle.dumps(self.hex_board, -1))
                if cell.is_empty():

                    if self.player == 1:
                        copy_board[i][j].value = [1, 0]
                    else:
                        copy_board[i][j].value = [0, 1]
                    children.append(HexStateManager(player=self.get_next_player(), dimensions=self.dimensions,
                                                    hex_board=copy_board, neighbours=self.neighbours))
        return children

    def generate_child_indexes(self):
        indexes = []

        for i, row in enumerate(self.hex_board):
            for j, cell in enumerate(row):
                if cell.is_empty():
                    indexes.append(1)
                else:
                    indexes.append(0)
        return indexes

    def is_terminal(self):
        player1 = self.check_player1_terminal()
        player2 = self.check_player2_terminal()

        return player1 or player2

    def check_player1_terminal(self):
        """
        Check if player 1 has reached a terminal state.
        Player 1 playes from left to right

        :return: True if terminal state reached, else False
        """
        unvisited = []
        visited = []

        for row in self.hex_board:
            if row[0].value == [1, 0]:  # Add first element of every row to unvisited
                unvisited.append(row[0])
        while len(unvisited):
            current_cell = unvisited[0]

            # Adds unvisited neighbours
            neighbours = self.neighbours.get(str(current_cell.position))
            for neighbour in neighbours:
                neighbour_object = self.hex_board[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [1, 0] and neighbour_object not in visited and neighbour_object not in \
                        unvisited:
                    unvisited.append(neighbour_object)

            # Add current cell to visited and remove from unvisited
            visited.append(unvisited.pop(0))

            # Checks if cell is on opposite side
            if current_cell in [x[-1] for x in self.hex_board]:
                self.winner = 1
                return True

        return False

    def check_player2_terminal(self):
        """
        Check if player 2 has reached a terminal state.
        Player 2 plays from right to left (aka. top to bottom)

        :return: True if terminal state reached, else False
        """
        unvisited = []
        visited = []

        for cell in self.hex_board[0]:
            if cell.value == [0, 1]:  # Add first element of every column to unvisited
                unvisited.append(cell)
        while len(unvisited):
            current_cell = unvisited[0]

            # Adds unvisited neighbours
            neighbours = self.neighbours.get(str(current_cell.position))
            for neighbour in neighbours:
                neighbour_object = self.hex_board[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [0, 1] and neighbour_object not in visited and neighbour_object not in \
                        unvisited:
                    unvisited.append(neighbour_object)

            # Add current cell to visited and remove from unvisited
            visited.append(unvisited.pop(0))

            # Checks if cell is on opposite side
            if current_cell in self.hex_board[-1]:
                self.winner = 2
                return True

        return False

    def get_winner(self):
        """
        Get the winner of the state if there is one

        :return: The winner
        """
        if self.winner is None:
            self.is_terminal()
        return self.winner

    def flatten_to_case(self, board):
        """
        Flatten the board state into a case for use in a neural network

        :param board: The board state to flatten
        :return: The case resulting from the board state
        """

        values = []
        flat_board = self.flatten(board)
        for cell in flat_board:
            values += [cell.get_player()]

        player = self.get_current_player()
        if player == 1:
            values += [1]
        elif player == 2:
            values += [2]
        return np.array(values)

    def print_state(self):
        """
        Print current state of the game
        """
        size = self.dimensions
        maxlen = len(str(size * size))
        m = size * 2 - 1
        matrix = [[' ' * maxlen] * m for _ in range(m)]

        board = self.flatten(self.hex_board)

        for n in range(size * size):
            r = n // size
            c = n % size
            value = self.get_hexcell_player(board[n].value)
            matrix[c + r][size - r - 1 + c] = '{0:{1}}'.format(value, maxlen)

        print('\n'.join(''.join(row) for row in matrix))
        print('')

    def flatten(self, board):
        """
        Flatten the current board state to a 1d array

        :return: 1d array of current board state
        """
        new_board = []
        for i in range(len(board)):
            for j in range(len(board)):
                state = board[i][j]
                new_board.append(state)
        return new_board

    def get_hexcell_player(self, cell_state):
        """
        Get the player id of the player occupying the cell

        :param cell_state: The state of the cell
        :return: The id of the player occupying the cell
        """
        if cell_state == [0, 0]:
            return "-"
        elif cell_state == [1, 0]:
            return "1"
        elif cell_state == [0, 1]:
            return "2"



