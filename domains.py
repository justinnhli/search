#!/usr/bin/env python3

from collections import defaultdict
from math import sqrt
from random import Random

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


class MazeState(GridWorldState):

    def __init__(self, num_rows, num_cols, seed=None):
        self.cells = {}
        for row in range(num_rows):
            for col in range(num_cols):
                self.cells[(row, col)] = set()
        self.rng = Random(seed)
        super().__init__(0, 0, num_rows, num_cols)
        self._create_maze()

    def _create_maze(self):
        remaining = set(self.cells.keys())
        # pick a random cell to include in the maze
        remaining.discard(self.rng.choice(sorted(remaining)))
        # do random walks from a random remaining cell
        while remaining:
            # pick a random remaining cell
            start = self.rng.choice(sorted(remaining))
            # random walk to a visited cell
            walk = {}
            cur = start
            while cur in remaining:
                direction = (0, 0)
                valid = False
                while not valid:
                    direction = self.rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                    nxt = (cur[0] + direction[0], cur[1] + direction[1])
                    valid = (0 <= nxt[0] < size and 0 <= nxt[1] < size)
                walk[cur] = direction
                cur = nxt
            # retrace from start and add cells to the maze
            cur = start
            while cur in walk:
                remaining.discard(cur)
                direction = walk[cur]
                self.cells[cur].add(direction)
                cur = (cur[0] + direction[0], cur[1] + direction[1])
                self.cells[cur].add((-1 * direction[0], -1 * direction[1]))

    def actions(self):
        actions = []
        cur = (self.row, self.col)
        for action in super().actions():
            direction = (action.row - self.row, action.col - self.col)
            if direction in self.cells[cur]:
                actions.append(action)
        return actions


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


class WordLadder(State):

    """A word ladder puzzle.

    This version does not currently support adding or removing letters.
    """

    WORDS_FILE_PATH = Path('/usr/share/dict').expanduser().resolve()
    WORDS = None

    def __init__(self, word):
        if WordLadder.WORDS is None:
            WordLadder.WORDS = defaultdict(set)
            with open(WordLadder.WORDS_FILE_PATH) as fd:
                for dict_word in fd:
                    dict_word = dict_word.lower().strip()
                    WordLadder.WORDS[len(dict_word)].add(dict_word)
        self.word = word

    def actions(self):
        actions = []
        for word in WordLadder.WORDS[len(self.word)]:
            if WordLadder.different_letters(self.word, word) == 1:
                actions.append(Action(
                    f'{self.word} -> {word}',
                    WordLadder(word),
                    1,
                ))
        return actions

    def heuristic(self, goal=None):
        return WordLadder.different_letters(self.word, goal.word)

    @staticmethod
    def different_letters(word1, word2):
        return (
            abs(len(word1) - len(word2))
            + sum(
                1 for char1, char2 in zip(word1, word2)
                if char1 != char2
            )
        )
