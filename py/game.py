import functools
import operator
from typing import Tuple

import numpy as np

from pieces import ALL_PIECES, ALL_PIECE_ORIENTATIONS, PieceOrientation

DEBUG_MODE = False
BOARD_SIZE = 20
NUM_PIECES = len(ALL_PIECES)
NUM_PIECE_ORIENTATIONS = sum([len(p.orientations) for p in ALL_PIECES])
NUM_MOVES_BOUND = NUM_PIECE_ORIENTATIONS * BOARD_SIZE * BOARD_SIZE

COLORS = ['R', 'G', 'B', 'Y']  # own the SW, NW, NE, and SE corners respectively
NUM_COLORS = len(COLORS)

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
    def __init__(self, piece_orientation: PieceOrientation, lower_left_corner: BoardLocation):
        self.piece_orientation = piece_orientation
        self.lower_left_corner = lower_left_corner
        self._id = to_move_index(piece_orientation.get_id(), lower_left_corner)
        self.name = f'{piece_orientation.name}@{board_location_code(lower_left_corner)}'

    def __eq__(self, other):
        return type(other) == Move and self._id == other.get_id()

    def __hash__(self):
        return self._id

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Move({self.name})'

    def get_id(self) -> MoveIndex:
        return self._id

    @staticmethod
    def from_id(index: MoveIndex) -> 'Move':
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
        self._occupancy_matrix = np.zeros((NUM_COLORS + 1, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self._occupancy_matrix[NUM_COLORS] = 1
        self._available_pieces = np.ones((NUM_COLORS, NUM_PIECES), dtype=bool)
        self._permissible_matrix = np.ones((NUM_COLORS, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self._required_matrix = np.zeros((NUM_COLORS, BOARD_SIZE, BOARD_SIZE), dtype=bool)

        # starting corners
        b = BOARD_SIZE - 1
        self._required_matrix[0][0][0] = 1
        self._required_matrix[1][0][b] = 1
        self._required_matrix[2][b][b] = 1
        self._required_matrix[3][b][0] = 1

        if DEBUG_MODE:
            self._validate()

    def _validate(self):
        assert np.sum(self._occupancy_matrix) == BOARD_SIZE * BOARD_SIZE
        assert np.all(functools.reduce(operator.or_, self._occupancy_matrix))
        for c, color in enumerate(COLORS):
            m = self._occupancy_matrix[c]

            available_piece_mask = self._available_pieces[c]
            used_piece_mask = ~available_piece_mask
            num_used_squares = sum([ALL_PIECES[i].size for i in np.where(used_piece_mask)[0]])
            assert num_used_squares == np.sum(m), (c, num_used_squares, np.sum(m))

            permissible_mask = self._occupancy_matrix[NUM_COLORS] & ~get_orthogonal_neighbors(m)
            assert np.all(permissible_mask == self._permissible_matrix[c])

            if num_used_squares:
                required_mask = permissible_mask & get_diagonal_neighbors(m)
                assert np.all(required_mask == self._required_matrix[c]), (color, mask_to_coordinates(required_mask), mask_to_coordinates(self._required_matrix[c]))
            else:
                assert np.sum(self._required_matrix[c]) == 1

    def to_ascii_drawing(self) -> str:
        """
        TODO: add pretty_print kwarg that prints out colored squares rather than R/G/B/Y
        """
        char_matrix = [list('%-2d ' % y) + ['.' for _ in range(BOARD_SIZE)] for y in range(BOARD_SIZE)]
        for (c, color) in enumerate(COLORS):
            for x, y in zip(*np.where(self._occupancy_matrix[c])):
                char_matrix[y][x+3] = color
        out_lines = list(reversed([''.join(char_list) for char_list in char_matrix]))
        out_lines.append('')
        out_lines.append(''.join([' ', ' ', ' '] + [chr(ord('A')+x) for x in range(BOARD_SIZE)]))
        return '\n'.join(out_lines)

    def get_legal_move_mask(self, c: ColorIndex) -> MoveMask:
        permissible_mask = self._permissible_matrix[c]
        required_coordinates = mask_to_coordinates(self._required_matrix[c])

        mask = np.zeros(NUM_MOVES_BOUND, dtype=bool)

        for piece_id in np.where(self._available_pieces[c])[0]:
            piece = ALL_PIECES[piece_id]
            for orientation in piece.orientations:
                for oi in range(piece.size):
                    ox, oy = orientation.coordinates[oi]
                    o_xy = np.array([ox, oy], dtype=int).reshape((1, 2))
                    for zx, zy in required_coordinates:
                        # test an orientation of the piece with (ox, oy) == (zx, zy)
                        z_xy = np.array([zx, zy], dtype=int).reshape((1, 2))
                        shift_xy = z_xy - o_xy
                        move_coordinates = orientation.coordinates + shift_xy
                        if np.min(move_coordinates) < 0 or np.max(move_coordinates) >= BOARD_SIZE:
                            continue
                        move_mask = coordinates_to_mask(move_coordinates)
                        if is_subset_of(move_mask, permissible_mask):
                            mask[to_move_index(orientation.get_id(), tuple(shift_xy.reshape((-1,))))] = 1

        return mask

    def apply(self, c: ColorIndex, move: Move):
        piece_orientation = move.piece_orientation
        lower_left_corner = move.lower_left_corner
        move_coordinates = piece_orientation.coordinates + lower_left_corner
        move_mask = coordinates_to_mask(move_coordinates)
        impermissible_mask = move_mask | get_orthogonal_neighbors(move_mask)
        self._occupancy_matrix[c] |= move_mask
        self._occupancy_matrix[NUM_COLORS] ^= move_mask
        self._available_pieces[c][piece_orientation.piece_id] = 0
        self._permissible_matrix[:] &= ~move_mask
        self._permissible_matrix[c] &= ~impermissible_mask
        self._required_matrix[c] &= ~impermissible_mask
        self._required_matrix[c] |= self._permissible_matrix[c] & get_diagonal_neighbors(move_mask)
        self._required_matrix[:] &= ~move_mask

        if DEBUG_MODE:
            self._validate()

    def announce_results(self):
        scores = []
        for c, color in enumerate(COLORS):
            available_piece_mask = self._available_pieces[c]
            score = sum([ALL_PIECES[i].size for i in np.where(available_piece_mask)[0]])
            scores.append(score)

        winning_score = min(scores)
        for color, score in zip(COLORS, scores):
            winner_str = ' (WINNER)' if score == winning_score else ''
            print(f'{color}: {score}{winner_str}')


def simulate_random_game(seed=123):
    np.random.seed(seed)
    state = GameState()

    move_made = True
    while move_made:
        print('')
        print(state.to_ascii_drawing())
        print('')
        move_made = False
        for c, color in enumerate(COLORS):
            mask = state.get_legal_move_mask(c)
            legal_moves = np.where(mask)[0]
            if len(legal_moves) == 0:
                print(f'{color}: PASS')
                continue
            move_made = True
            move_index = np.random.choice(legal_moves)
            move = Move.from_id(move_index)
            print(f'{color}: {move}')
            state.apply(c, move)

    print('')
    state.announce_results()


if __name__ == '__main__':
    simulate_random_game()
