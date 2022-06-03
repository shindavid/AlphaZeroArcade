import itertools
from typing import List

import numpy as np


"""
A PieceOrientationCoordinates corresponds to a particular orientation of a Piece.

It is represented by an (N, 2)-shaped int ndarray, where N is the number of squares in the piece.
Each row of the array corresponds to (x, y) coordinates of grid squares occupied by the piece.

The representation is normalized by:

- shifting to border the x/y-axes within the 1st quadrant
- ordering the array rows lexicographically

An RawPieceOrientationCoordinates is an un-normalized PieceOrientationCoordinates.
"""
PieceOrientationCoordinates = np.ndarray
RawPieceOrientationCoordinates = np.ndarray


def normalize(c: RawPieceOrientationCoordinates) -> PieceOrientationCoordinates:
    """
    Shifts to border the x/y-axes within the 1st quadrant.
    Orders the array rows lexicographically.
    """
    c = c[np.lexsort(c.T[::-1])]  # https://stackoverflow.com/a/38277186/543913
    c[:, 0] -= min(c[:, 0])
    c[:, 1] -= min(c[:, 1])
    return c


def rotate_clockwise(c: PieceOrientationCoordinates, n: int = 1) -> PieceOrientationCoordinates:
    """
    Rotates the given PieceOrientationCoordinates 90*n degrees clockwise, and returns it.
    """
    n = n % 4
    xs = (+1, +1, -1, -1)[n]
    ys = (+1, -1, -1, +1)[n]
    xi = n % 2
    yi = 1 - xi
    x = xs * c[:, xi].reshape((-1, 1))
    y = ys * c[:, yi].reshape((-1, 1))
    return normalize(np.hstack((x, y)))


def reflect_over_x_axis(c: PieceOrientationCoordinates) -> PieceOrientationCoordinates:
    """
    Reflects the given PieceOrientationCoordinates over the x-axis, and returns it.
    """
    x = c[:, 0].reshape((-1, 1))
    y = c[:, 1].reshape((-1, 1))
    return normalize(np.hstack((x, -y)))


def compute_piece_orientation_coordinates(ascii_drawing: str) -> PieceOrientationCoordinates:
    """
    ascii_drawing consists of one or more lines of text, each consisting of ' ' or 'x' characters,
    with the 'x' characters corresponding to occupied spaces.
    """
    lines = list(reversed(ascii_drawing.splitlines()))
    coordinates = []
    for y, line in enumerate(lines):
        for x, c in enumerate(line):
            if c == ' ':
                continue
            if c != 'x':
                raise Exception(f'Bad ascii_drawing: {ascii_drawing}')
            coordinates.append([x, y])
    assert coordinates, f'Bad ascii_drawing: {ascii_drawing}'
    return normalize(np.array(coordinates))


def block_str_join(strs: List[str], delim: str) -> str:
    """
    Horizontally concatenates the multi-line strings in strs, aligning along the bottom, with
    vertically-stretched copies of delim separating them.

    Returns the multi-line string formed in this way.
    """
    lines_list = [s.splitlines() for s in strs]
    max_y = max(map(len, lines_list))
    max_x_list = [max(map(len, lines)) for lines in lines_list]

    for i, (lines, max_x) in enumerate(zip(lines_list, max_x_list)):
        blank_lines = [''] * (max_y - len(lines))
        lines_list[i] = blank_lines + lines

    fmt_strs = [f'%-{x}s' for x in max_x_list]
    out = [delim.join([fmt % s for (fmt, s) in zip(fmt_strs, lines)]) for lines in zip(*lines_list)]
    return '\n'.join(out)


class PieceOrientation:
    _next_index = 0

    def __init__(self, name: str, piece_index: int, coordinates: PieceOrientationCoordinates):
        self.index = PieceOrientation._next_index
        PieceOrientation._next_index += 1
        self.piece_index = piece_index
        self.name = name
        self.coordinates = coordinates

    def __eq__(self, other):
        return type(other) == PieceOrientation and self.index == other.index

    def __hash__(self):
        return self.index

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'PieceOrientation({self.name})'

    def get_ascii_drawing(self) -> str:
        coordinates = self.coordinates

        max_x = max(coordinates[:, 0])
        max_y = max(coordinates[:, 1])

        char_matrix = [[' '] * (max_x + 1) for _ in range(max_y + 1)]
        for x, y in coordinates:
            char_matrix[y][x] = 'x'

        return '\n'.join([''.join(c) for c in reversed(char_matrix)] + [self.name])


def get_rank_key(coordinates: PieceOrientationCoordinates):
    """
    Returns a sorted-tuple of all (x, y) pairs in coordinates.

    This is used to define the notion of the canonical orientation of a PieceOrientationCoordinates.
    """
    return tuple(map(tuple, coordinates))


class Piece:
    _next_index = 0

    def __init__(self, name: str, ascii_drawing: str):
        """
        ascii_drawing can be in any orientation. The constructor normalized appropriately.
        """
        self.index = Piece._next_index
        Piece._next_index += 1
        self.name = name
        assert len(name) == 2 and name[1] in '12345', name
        self.size = int(name[1])
        self.orientations: List[PieceOrientation] = []

        coordinates = compute_piece_orientation_coordinates(ascii_drawing)

        # first compute canonical
        coordinates_dict = {}
        for r, m in [('r', coordinates), ('R', reflect_over_x_axis(coordinates))]:
            for n in range(4):
                m2 = rotate_clockwise(m, n)
                coordinates_dict[m2.tobytes()] = m2

        canonical_coordinates = list(sorted(coordinates_dict.values(), key=get_rank_key))[0]

        # now compute orientations relative to canonical
        oset = set()
        for r, m in [('r', canonical_coordinates), ('R', reflect_over_x_axis(canonical_coordinates))]:
            for n in range(4):
                m2 = rotate_clockwise(m, n)
                key = m2.tobytes()
                if key not in oset:
                    descr = f'{name}{r}{n}'
                    self.orientations.append(PieceOrientation(descr, self.index, m2))
                    oset.add(key)

        self._validate()

    def verbose_repr(self) -> str:
        repr_list = [self.name] + [o.get_ascii_drawing() for o in self.orientations]
        return block_str_join(repr_list, ' | ')

    @property
    def canonical_orientation(self) -> PieceOrientation:
        return self.orientations[0]

    def __eq__(self, other):
        return type(other) == Piece and self.index == other.index

    def __hash__(self):
        return self.index

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Piece({self.name})'

    def _validate(self):
        for orientation in self.orientations:
            assert len(orientation.coordinates) == self.size, (self.name, orientation, self.size)
        assert len(self.orientations) in (1, 2, 4, 8), (self.name, len(self.orientations))


O1 = Piece('O1', 'x')
I2 = Piece('I2', 'xx')
I3 = Piece('I3', 'xxx')
L3 = Piece('L3', """
xx
x
""")
I4 = Piece('I4', 'xxxx')
O4 = Piece('O4', """
xx
xx
""")
T4 = Piece('T4', """
xxx
 x
""")
L4 = Piece('L4', """
xxx
x
""")
S4 = Piece('S4', """
xx
 xx
""")
F5 = Piece('F5', """
 xx
xx
 x
""")
I5 = Piece('I5', 'xxxxx')
L5 = Piece('L5', """
xxxx
x
""")
N5 = Piece('N5', """
xx
 xxx
""")
P5 = Piece('P5', """
xx
xxx
""")
T5 = Piece('T5', """
xxx
 x
 x
""")
U5 = Piece('U5', """
x x
xxx
""")
V5 = Piece('V5', """
xxx
x
x
""")
W5 = Piece('W5', """
x
xx
 xx
""")
X5 = Piece('X5', """
 x
xxx
 x
""")
Y5 = Piece('Y5', """
xxxx
 x
""")
Z5 = Piece('Z5', """
xx
 x
 xx
""")

ALL_PIECES = [
    O1,
    I2,
    I3, L3,
    I4, O4, T4, L4, S4,
    F5, I5, L5, N5, P5, T5, U5, V5, W5, X5, Y5, Z5
]

ALL_PIECE_ORIENTATIONS = list(itertools.chain(*[piece.orientations for piece in ALL_PIECES]))


if __name__ == '__main__':
    assert len(ALL_PIECES) == len(set(ALL_PIECES))
    for piece in ALL_PIECES:
        print('----------------------------------------------------------')
        print(piece.verbose_repr())
