#!/usr/bin/env python3

from math import sqrt

from search import State, Action


class MissionariesCannibalsState(State):

    def __init__(self, missionaries, cannibals, boat):
        super().__init__()
        self.missionaries = missionaries
        self.cannibals = cannibals
        self.boat = boat

    def actions(self):
        actions = []
        # calculate departing side people
        if self.boat == 1:
            num_m = self.missionaries
            num_c = self.cannibals
        else:
            num_m = 3 - self.missionaries
            num_c = 3 - self.cannibals
        # for each possible passenger combination
        for m, c in ((0, 1), (0, 2), (1, 1), (1, 0), (2, 0)):
            # calculate the new arrangement
            if m > num_m or c > num_c:
                continue
            if self.boat == 1:
                new_m = num_m - m
                new_c = num_c - c
            else:
                new_m = self.missionaries + m
                new_c = self.cannibals + c
            new_b = 1 - self.boat
            # if the new arrangement fails the puzzle, don't add it
            if new_c > new_m and new_m > 0:
                continue
            if (3 - new_c) > (3 - new_m) and (3 - new_m) > 0:
                continue
            # otherwise, store the action
            action = Action(
                    '{}M, {}C'.format(m, c),
                    MissionariesCannibalsState(new_m, new_c, new_b),
                    1)
            actions.append(action)
        return actions


class GridWorldState(State):

    def __init__(self, row, col, num_rows, num_cols):
        self.row = row
        self.col = col
        self.num_rows = num_rows
        self.num_cols = num_cols

    def actions(self):
        actions = []
        if self.col - 1 >= 0:
            actions.append(Action(
                'left',
                GridWorldState(self.row, self.col-1, self.num_rows, self.num_cols),
                1))
        if self.row - 1 >= 0:
            actions.append(Action(
                'down',
                GridWorldState(self.row-1, self.col, self.num_rows, self.num_cols),
                1))
        if self.col + 1 < self.num_cols:
            actions.append(Action(
                'right',
                GridWorldState(self.row, self.col+1, self.num_rows, self.num_cols),
                1))
        if self.row + 1 < self.num_rows:
            actions.append(Action(
                'up',
                GridWorldState(self.row+1, self.col, self.num_rows, self.num_cols),
                1))
        return actions

    def heuristic(self, goal=None):
        return sqrt((self.row - goal.row)**2 + (self.col - goal.col)**2)


class PolynomialDescentState(State):

    def __init__(self, x):
        self.x = x

    def actions(self):
        actions = []
        actions.append(Action(
            '+1',
            PolynomialDescentState(self.x + 1),
            0))
        actions.append(Action(
            '-1',
            PolynomialDescentState(self.x - 1),
            0))
        return actions

    def heuristic(self, goal=None):
        return (self.x + 1)**2


class SlidingPuzzleState(State):

    """The state of a sliding puzzle of size 3 or 4

    The goal of this puzzle is to arrange the tiles such that the blank tile
    is on the upper left. For a puzzle of size 3, the goal state will be:

      1 2
    3 4 5
    6 7 8

    The state is represented as a single string, with the tiles read off left
    to right, top to bottom. The above goal state is therefore represented as
    " 12345678".
    """

    def __init__(self, state_str, width=None, height=None):
        """Constructor

        Arguments:
            state_str (str): The board, as a string. Must either 9 or 16
                characters long, and contain the appropriate characters from
                " 123456789ABCDEF" exactly once.
            width (int, optional): The width of the board. By default, assumes
                the board is square.
            height (int, optional): The width of the board. By default, assumes
                the board is square.
        """
        self._check_state_str(state_str)
        self.state = state_str
        self._width, self._height = self._calculate_size(width, height)

    def _check_state_str(self, state_str):
        VALID_CHARS = ' '
        VALID_CHARS += '123456789'
        VALID_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        VALID_CHARS += 'abcdefghijklmnopqrstuvwxyz'
        if len(set(state_str)) != len(state_str):
            raise ValueError('State string must contain unique characters')
        if len(set(state_str).intersection(set(VALID_CHARS))) != len(state_str):
            raise ValueError('State string must uniquely contain "{}"'.format(VALID_CHARS))

    def _calculate_size(self, width, height):
        if width is None and height is None:
            width = int(sqrt(len(self.state)))
            height = int(sqrt(len(self.state)))
        elif width is None:
            height = int(height)
            width = int(len(self.state) / height)
        elif height is None:
            width = int(width)
            height = int(len(self.state) / width)
        else:
            width = int(width)
            height = int(height)
        if width * height != len(self.state):
            raise ValueError('Given width/height is not valid for given state string')
        return width, height

    def in_bounds(self, row, col):
        """Check if a particular row and column location are on the board

        Arguments:
            row (int): The row
            col (int): The column

        Returns:
            bool: True if the location is on the board
        """
        return (0 <= row < self._height and 0 <= col < self._width)

    def coord_to_index(self, row, col):
        """Convert a row and column to an index

        Arguments:
            row (int): The row
            col (int): The column

        Returns:
            int: The index of the specified location in the state string
        """
        return row * self._width + col

    def index_to_coord(self, index):
        """Convert a state string index to a row and column

        Arguments:
            index (int): The index

        Returns:
            [int, int]: The row and column, in a list
        """
        return [index // self._width, index % self._width]

    def actions(self):
        """Get the possible actions from this state

        Returns:
            [Action]: The list of possible actions
        """
        index = self.state.index(' ')
        row, col = self.index_to_coord(index)
        result = []
        # dr and dc are the row-step and column-step respectively
        for name, dr, dc in (('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)):
            if self.in_bounds(row + dr, col + dc):
                next_state = list(self.state)
                swap_index = self.coord_to_index(row + dr, col + dc)
                next_state[index] = next_state[swap_index]
                next_state[swap_index] = ' '
                next_state = ''.join(next_state)
                result.append(Action(name, SlidingPuzzleState(next_state), 1))
        return result

    def heuristic(self, goal=None):
        """Calculate the heuristic from the current state

        Returns:
            int: The (under-)estimated number of moves to reach the goal
        """
        total = 0
        for tile in goal.state[:-1]:
            goal_row, goal_col = self.index_to_coord(goal.state.index(tile))
            row, col = self.index_to_coord(self.state.index(tile))
            total += abs(goal_row - row) + abs(goal_col - col)
        return total
