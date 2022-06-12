import functools
import operator
from typing import Tuple, Optional, List

import numpy as np
from termcolor import colored

from pieces import ALL_PIECES, ALL_PIECE_ORIENTATIONS, PieceOrientation

DEBUG_MODE = False
BOARD_SIZE = 20
NUM_PIECES = len(ALL_PIECES)
NUM_PIECE_ORIENTATIONS = sum([len(p.orientations) for p in ALL_PIECES])
NUM_MOVES_BOUND = NUM_PIECE_ORIENTATIONS * BOARD_SIZE * BOARD_SIZE + 1  # +1 for pass move

COLORS = ['R', 'G', 'B', 'Y']  # own the SW, NW, NE, and SE corners respectively
NUM_COLORS = len(COLORS)
SQUARE_CHAR = chr(9607) * 2
PRETTY_COLORS = [colored(SQUARE_CHAR, c) for c in ('red', 'green', 'blue', 'yellow')]

Score = int
ColorIndex = int
MoveIndex = int
PieceIndex = int
PieceOrientationIndex = int
BoardMask = np.ndarray  # (BOARD_SIZE, BOARD_SIZE) bool
BoardCoordinates = np.ndarray  # (n, 2) int, each row is (x, y)
BoardLocation = Tuple[int, int]
MoveMask = np.ndarray  # NUM_MOVES_BOUND bool


def coordinates_to_mask(coordinates: BoardCoordinates) -> BoardMask:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    np.ravel(board)[np.ravel_multi_index(coordinates.T, board.shape)] = 1  # https://stackoverflow.com/a/46018613/543913
    return board


def mask_to_coordinates(mask: BoardMask) -> BoardCoordinates:
    x, y = np.where(mask)
    return np.vstack((x, y)).T


def mask_shift(mask: BoardMask, delta_x: int, delta_y: int) -> BoardMask:
    coordinates = mask_to_coordinates(mask)
    coordinates[:, 0] += delta_x
    coordinates[:, 1] += delta_y
    in_bounds_x = (0 <= coordinates.T[0]) & (coordinates.T[0] < BOARD_SIZE)
    in_bounds_y = (0 <= coordinates.T[1]) & (coordinates.T[1] < BOARD_SIZE)
    return coordinates_to_mask(coordinates[np.where(in_bounds_x & in_bounds_y)[0]])


def get_orthogonal_neighbors(mask: BoardMask) -> BoardMask:
    mask_n = mask_shift(mask, 0, +1)
    mask_w = mask_shift(mask, -1, 0)
    mask_s = mask_shift(mask, 0, -1)
    mask_e = mask_shift(mask, +1, 0)
    return (mask_n | mask_w | mask_s | mask_e) & ~mask


def get_diagonal_neighbors(mask: BoardMask) -> BoardMask:
    mask_nw = mask_shift(mask, -1, +1)
    mask_ne = mask_shift(mask, +1, +1)
    mask_se = mask_shift(mask, +1, -1)
    mask_sw = mask_shift(mask, -1, -1)
    return (mask_nw | mask_ne | mask_se | mask_sw) & ~mask


def board_location_code(loc: BoardLocation) -> str:
    x = chr(ord('A') + loc[0])
    y = str(loc[1])
    return x + y


def to_move_index(piece_orientation_index: PieceOrientationIndex, lower_left_corner: BoardLocation) -> MoveIndex:
    x, y = lower_left_corner
    return piece_orientation_index * BOARD_SIZE * BOARD_SIZE + x * BOARD_SIZE + y


class Move:
    def __init__(self, piece_orientation: Optional[PieceOrientation], lower_left_corner: BoardLocation):
        self.piece_orientation = piece_orientation
        self.lower_left_corner = lower_left_corner
        if piece_orientation is None:
            self.index = NUM_MOVES_BOUND - 1
            self.name = 'pass'
        else:
            self.index = to_move_index(piece_orientation.index, lower_left_corner)
            self.name = f'{piece_orientation.name}@{board_location_code(lower_left_corner)}'

    def is_pass(self):
        return self.index == NUM_MOVES_BOUND - 1

    def __eq__(self, other):
        return type(other) == Move and self.index == other.index

    def __hash__(self):
        return self.index

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Move({self.name})'

    @staticmethod
    def get_pass() -> 'Move':
        return Move(None, (0, 0))

    @staticmethod
    def from_index(index: MoveIndex) -> 'Move':
        if index == NUM_MOVES_BOUND - 1:
            return Move.get_pass()
        piece_orientation_index = index // (BOARD_SIZE * BOARD_SIZE)
        piece_orientation = ALL_PIECE_ORIENTATIONS[piece_orientation_index]
        loc_value = index % (BOARD_SIZE * BOARD_SIZE)
        x = loc_value // BOARD_SIZE
        y = loc_value % BOARD_SIZE
        loc = (x, y)
        return Move(piece_orientation, loc)


def is_subset_of(sub_mask: BoardMask, super_mask: BoardMask) -> bool:
    return np.array_equal(sub_mask | super_mask, super_mask)


class GameState:
    def __init__(self):
        """
        * _occupancy_matrix: which color occupies which square. Last row corresponds to the unoccupied "color"
        * _available_pieces: which pieces have yet to be placed, per color
        * _permissible_matrix: for each color, which unoccupied squares are not adjacent to that color
        * _required_matrix: for each color, the set of permissible squares that the next move must intersect
        """
        self.occupancy_matrix = np.zeros((NUM_COLORS + 1, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self.available_pieces = np.ones((NUM_COLORS, NUM_PIECES), dtype=bool)
        self.permissible_matrix = np.ones((NUM_COLORS, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self.required_matrix = np.zeros((NUM_COLORS, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self.current_color_array = np.zeros(NUM_COLORS, dtype=bool)

        # 0 is the starting player
        self.current_color_array[0] = 1

        # all squares are initially unoccupied
        self.occupancy_matrix[NUM_COLORS] = 1

        # starting moves must occupy the corners
        b = BOARD_SIZE - 1
        self.required_matrix[0][0][0] = 1
        self.required_matrix[1][0][b] = 1
        self.required_matrix[2][b][b] = 1
        self.required_matrix[3][b][0] = 1

        if DEBUG_MODE:
            self._validate()

    @property
    def unoccupied_matrix(self) -> BoardMask:
        return self.occupancy_matrix[NUM_COLORS]

    def copy_from(self, state: 'GameState'):
        self.occupancy_matrix = np.copy(state.occupancy_matrix)
        self.available_pieces = np.copy(state.available_pieces)
        self.permissible_matrix = np.copy(state.permissible_matrix)
        self.required_matrix = np.copy(state.required_matrix)
        self.current_color_array = np.copy(state.current_color_array)

    def get_current_color_index(self) -> ColorIndex:
        return np.where(self.current_color_array)[0][0]

    def _validate(self):
        assert np.sum(self.occupancy_matrix) == BOARD_SIZE * BOARD_SIZE
        assert np.all(functools.reduce(operator.or_, self.occupancy_matrix))
        for c, color in enumerate(COLORS):
            m = self.occupancy_matrix[c]

            available_piece_mask = self.available_pieces[c]
            used_piece_mask = ~available_piece_mask
            num_used_squares = sum([ALL_PIECES[i].size for i in np.where(used_piece_mask)[0]])
            assert num_used_squares == np.sum(m), (c, num_used_squares, np.sum(m))

            permissible_mask = self.occupancy_matrix[NUM_COLORS] & ~get_orthogonal_neighbors(m)
            assert np.all(permissible_mask == self.permissible_matrix[c])

            if num_used_squares:
                required_mask = permissible_mask & get_diagonal_neighbors(m)
                assert np.all(required_mask == self.required_matrix[c]), (color, mask_to_coordinates(required_mask), mask_to_coordinates(self.required_matrix[c]))
            else:
                assert np.sum(self.required_matrix[c]) == 1

    def to_ascii_drawing(self, pretty_print=True) -> str:
        colors = PRETTY_COLORS if pretty_print else COLORS
        p = ' ' if pretty_print else ''
        empty_color = '  ' if pretty_print else '.'
        char_matrix = [list('%-2d ' % y) + [empty_color for _ in range(BOARD_SIZE)] for y in range(BOARD_SIZE)]
        for (c, color) in enumerate(colors):
            for x, y in zip(*np.where(self.occupancy_matrix[c])):
                char_matrix[y][x+3] = color
        out_lines = list(reversed([''.join(char_list) for char_list in char_matrix]))
        out_lines.append('')
        out_lines.append(''.join([' ', ' ', ' '] + [chr(ord('A')+x) + p for x in range(BOARD_SIZE)]))
        return '\n'.join(out_lines)

    def apply_move(self, c: ColorIndex, move: Move):
        assert self.current_color_array[c] and sum(self.current_color_array) == 1
        self.current_color_array[c] = 0
        self.current_color_array[(c+1) % NUM_COLORS] = 1
        if move.is_pass():
            return

        piece_orientation = move.piece_orientation
        lower_left_corner = move.lower_left_corner
        move_coordinates = piece_orientation.coordinates + lower_left_corner
        move_mask = coordinates_to_mask(move_coordinates)
        impermissible_mask = move_mask | get_orthogonal_neighbors(move_mask)

        assert np.all(self.permissible_matrix[c] & move_mask == move_mask)
        assert np.any(self.required_matrix[c] & move_mask)

        self.occupancy_matrix[c] |= move_mask
        self.occupancy_matrix[NUM_COLORS] ^= move_mask
        self.available_pieces[c][piece_orientation.piece_index] = 0
        self.permissible_matrix[:] &= ~move_mask
        self.permissible_matrix[c] &= ~impermissible_mask
        self.required_matrix[c] &= ~impermissible_mask
        self.required_matrix[c] |= self.permissible_matrix[c] & get_diagonal_neighbors(move_mask)
        self.required_matrix[:] &= ~move_mask

        if DEBUG_MODE:
            self._validate()

    def get_scores(self) -> List[Score]:
        scores = []
        for c, color in enumerate(COLORS):
            available_piece_mask = self.available_pieces[c]
            score = sum([ALL_PIECES[i].size for i in np.where(available_piece_mask)[0]])
            scores.append(score)

        return scores


class TuiGameManager:
    def __init__(self, players: List, pretty_print: bool = True):
        n = len(players)
        assert n in (2, 4)
        self.num_players = n
        self.players = list(players)
        np.random.shuffle(self.players)  # randomize starting order
        self.color_to_player = self.players
        if n == 4:
            for c, player in enumerate(self.players):
                player.receive_color_assignment(c)
        elif n == 2:
            self.color_to_player = self.players + self.players
            for c, player in enumerate(self.players):
                player.receive_color_assignment(c, c+2)

        self.pretty_print = pretty_print

    def run(self, silent: bool = False):
        if not silent:
            if self.num_players == 4:
                for color, player in zip(COLORS, self.players):
                    print(f'{color} {player}')
            else:
                for c, player in enumerate(self.players):
                    color1 = COLORS[c]
                    color2 = COLORS[c+2]
                    print(f'{color1}{color2} {player}')

        state = GameState()
        all_passed = False
        while not all_passed:
            if not silent:
                print('')
                print(state.to_ascii_drawing(self.pretty_print))
                print('')
            all_passed = True
            for c, player in enumerate(self.color_to_player):
                color = COLORS[c]
                move = player.get_move(state)
                all_passed &= move.is_pass()
                if not silent:
                    print(f'{color}: {move}')
                state.apply_move(c, move)

        color_scores = state.get_scores()
        player_scores = [sum([color_scores[c] for c in player.color_indices]) for player in self.players]
        winning_score = min(player_scores)
        winning_players = [p for p, s in zip(self.players, player_scores) if s == winning_score]

        if not silent:
            print('')
            for player, score in zip(self.players, player_scores):
                won = score == winning_score
                winner_str = ' (WINNER)' if won else ''
                color_str = '+'.join([COLORS[c] for c in player.color_indices])
                score_str = '+'.join([str(color_scores[c]) for c in player.color_indices])
                print(f'{color_str} {player}: {score_str}{winner_str}')

        return winning_players
