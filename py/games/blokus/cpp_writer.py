from games.blokus.pieces import ALL_PIECES, ALL_PIECE_ORIENTATIONS, Piece, PieceOrientation

from collections import defaultdict
from typing import Iterable, List

import numpy as np


def extract_row_masks(matrix):
    masks = []

    for row in matrix.T:
        mask = 0
        for b in reversed(row):
            mask = (mask << 1) | b
        masks.append(mask)
    return masks


class PieceData:
    def __init__(self, piece: Piece):
        self.piece = piece
        self.subrange_lengths = [0, 0, 0]
        self.constraints_dict = defaultdict(list)
        self.corner_range_start = None

    def add_constraint(self, constraint, corner_index):
        self.constraints_dict[constraint].append(corner_index)

    def finalize(self, start):
        size1 = set()
        for c in range(1, 5):
            size1.add(len(self.constraints_dict.get(c, [])))
        assert len(size1) == 1, size1

        size2 = set()
        for c in range(5, 9):
            size2.add(len(self.constraints_dict.get(c, [])))
        assert len(size2) == 1, size2

        self.subrange_lengths[0] = len(self.constraints_dict.get(0, []))
        self.subrange_lengths[1] = len(self.constraints_dict.get(1, []))
        self.subrange_lengths[2] = len(self.constraints_dict.get(5, []))

        self.corner_range_start = start


piece_data_list = []
for piece in ALL_PIECES:
    piece_data_list.append(PieceData(piece))


class PieceOrientationData:
    def __init__(self, piece_orientation: PieceOrientation, mask_array_start_index):
        self.piece_orientation = piece_orientation
        self.drawing = piece_orientation.get_ascii_drawing()

        self.width = max(piece_orientation.coordinates[:, 0]) + 1
        self.height = max(piece_orientation.coordinates[:, 1]) + 1

        self.mask_array_start_index = mask_array_start_index

        region = np.zeros((self.width, self.height), dtype=bool)
        shifted_region = np.zeros((self.width + 2, self.height + 2), dtype=bool)
        adjacency_region = np.zeros((self.width + 2, self.height + 2), dtype=bool)
        diagonal_region = np.zeros((self.width + 2, self.height + 2), dtype=bool)

        for (x, y) in piece_orientation.coordinates:
            region[x, y] = 1
            shifted_region[x + 1, y + 1] = 1
            adjacency_region[x + 1, y] = 1
            adjacency_region[x + 1, y + 2] = 1
            adjacency_region[x, y + 1] = 1
            adjacency_region[x + 2, y + 1] = 1
            diagonal_region[x, y] = 1
            diagonal_region[x + 2, y] = 1
            diagonal_region[x, y + 2] = 1
            diagonal_region[x + 2, y + 2] = 1

        adjacency_region &= ~shifted_region
        diagonal_region &= ~shifted_region
        diagonal_region &= ~adjacency_region

        self.shifted_region = shifted_region
        self.adjacency_region = adjacency_region
        self.diagonal_region = diagonal_region

        self.region_xy_list = list(zip(*np.where(region)))
        self.adjacency_xy_list = list(zip(*np.where(adjacency_region)))
        self.diagonal_xy_list = list(zip(*np.where(diagonal_region)))

        self.row_masks = extract_row_masks(shifted_region)[1:-1]
        self.adjacency_masks = extract_row_masks(adjacency_region)
        self.diagonal_masks = extract_row_masks(diagonal_region)

        self.corner_xy_list = []
        for (x, y) in piece_orientation.coordinates:
            x += 1
            y += 1
            if shifted_region[x, y-1] and shifted_region[x, y+1]:
                continue
            if shifted_region[x+1, y] and shifted_region[x-1, y]:
                continue
            self.corner_xy_list.append((x, y))

        assert len(self.row_masks) == self.height
        assert len(self.adjacency_masks) == self.height + 2
        assert len(self.diagonal_masks) == self.height + 2

    def get_ascii_drawing(self, xy=None, exclude_boundary=False):
        if exclude_boundary:
            char_matrix = [[' '] * self.width for _ in range(self.height)]
            for x, y in self.region_xy_list:
                char_matrix[y][x] = 'o'

            if xy is not None:
                char_matrix[xy[1] - 1][xy[0] - 1] = 'x'

            return '\n'.join(''.join(c).rstrip() for c in reversed(char_matrix))

        char_matrix = [[' '] * (self.width + 2) for _ in range(self.height + 2)]
        for x, y in self.region_xy_list:
            char_matrix[y+1][x+1] = 'o'
        for x, y in self.adjacency_xy_list:
            char_matrix[y][x] = '.'
        for x, y in self.diagonal_xy_list:
            char_matrix[y][x] = '*'

        if xy is not None:
            char_matrix[xy[1]][xy[0]] = 'x'

        return '\n'.join(''.join(c) for c in reversed(char_matrix))


kPieceOrientationRowMasks = []

piece_orientation_data_list = []
for piece_orientation in ALL_PIECE_ORIENTATIONS:
    i = len(kPieceOrientationRowMasks)
    data = PieceOrientationData(piece_orientation, i)
    piece_orientation_data_list.append(data)

    kPieceOrientationRowMasks.extend(data.row_masks)
    kPieceOrientationRowMasks.extend(data.adjacency_masks)
    kPieceOrientationRowMasks.extend(data.diagonal_masks)


ccNone = 0
ccN = 1
ccE = 2
ccS = 3
ccW = 4
ccNE = 5
ccSE = 6
ccSW = 7
ccNW = 8


NESW_OCCUPANCY_DICT = {
    (0, 0, 0, 0): list(range(9)),
    (1, 0, 0, 0): [ccNW, ccN, ccNE],
    (0, 1, 0, 0): [ccNE, ccE, ccSE],
    (0, 0, 1, 0): [ccSE, ccS, ccSW],
    (0, 0, 0, 1): [ccSW, ccW, ccNW],
    (1, 1, 0, 0): [ccNE],
    (0, 1, 1, 0): [ccSE],
    (0, 0, 1, 1): [ccSW],
    (1, 0, 0, 1): [ccNW],
    }


class PieceOrientationCornerData:
    def __init__(self, index, x, y, piece_index, piece_orientation_index):
        self.x = x
        self.y = y
        self.piece_index = piece_index
        self.piece_orientation_index = piece_orientation_index

        orientation_data = piece_orientation_data_list[piece_orientation_index]
        N_occupied = int(orientation_data.shifted_region[x, y+1])
        E_occupied = int(orientation_data.shifted_region[x+1, y])
        S_occupied = int(orientation_data.shifted_region[x, y-1])
        W_occupied = int(orientation_data.shifted_region[x-1, y])

        NESW = (N_occupied, E_occupied, S_occupied, W_occupied)
        self.compatible_constraints = NESW_OCCUPANCY_DICT[NESW]


piece_orientation_corner_data_list = []
for piece_orientation_data in piece_orientation_data_list:
    piece_index = piece_orientation_data.piece_orientation.piece_index
    piece_data = piece_data_list[piece_index]
    piece_orientation_index = piece_orientation_data.piece_orientation.index
    for x, y in piece_orientation_data.corner_xy_list:
        index = len(piece_orientation_corner_data_list)
        corner_data = PieceOrientationCornerData(index, x, y, piece_index, piece_orientation_index)
        piece_orientation_corner_data_list.append(corner_data)

        for c in corner_data.compatible_constraints:
            piece_data.add_constraint(c, index)


kCornerConstraintArray = []
for piece_data in piece_data_list:
    piece_data.finalize(len(kCornerConstraintArray))
    for c in range(9):
        kCornerConstraintArray.extend(piece_data.constraints_dict.get(c, []))


print('// Auto-generated by py/games/blokus/cpp_writer.py')

header = """
#include <games/blokus/Constants.hpp>
#include <games/blokus/Types.hpp>

namespace blokus {
namespace tables {
"""
print(header)
print('const _PieceData kPieceData[kNumPieces] = {')
for p, piece_data in enumerate(piece_data_list):
    piece = piece_data.piece
    name = piece.name
    end = ',' if p < len(piece_data_list) - 1 else ''
    canonical_orientation = piece.orientations[0]

    drawing = canonical_orientation.get_ascii_drawing()
    drawing_lines = drawing.splitlines()[:-1]

    if p > 0:
        print('')
    print(f'  // {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    subrange_lengths = piece_data.subrange_lengths
    corner_range_start = piece_data.corner_range_start

    subrange_lengths_str = '{%s}' % (', '.join(map(str, subrange_lengths)))
    print(f'  {{"{name}", {subrange_lengths_str}, {corner_range_start}}}{end}')

print('};  // kPieceData')
print('')

print('const _PieceOrientationData kPieceOrientationData[kNumPieceOrientations] = {')
for o, piece_orientation_data in enumerate(piece_orientation_data_list):
    orientation = piece_orientation_data.piece_orientation
    sym_index = orientation.sym_index
    piece = ALL_PIECES[orientation.piece_index]
    name = f'{piece.name}/{sym_index}'
    end = ',' if o < len(piece_orientation_data_list) - 1 else ''

    drawing = piece_orientation_data.get_ascii_drawing()
    drawing_lines = drawing.splitlines()

    if o > 0:
        print('')
    print(f'  // {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    mask_array_start_index = piece_orientation_data.mask_array_start_index
    width = piece_orientation_data.width
    height = piece_orientation_data.height
    print(f'  {{{mask_array_start_index}, {width}, {height}}}{end}')

print('};  // kPieceOrientationData')
print('')

print('const _PieceOrientationCornerData kPieceOrientationCornerData[kNumPieceOrientationCorners] = {')
for c, piece_orientation_corner_data in enumerate(piece_orientation_corner_data_list):
    x = piece_orientation_corner_data.x
    y = piece_orientation_corner_data.y
    p = piece_orientation_corner_data.piece_index
    po = piece_orientation_corner_data.piece_orientation_index

    end = ',' if c < len(piece_orientation_corner_data_list) - 1 else ''

    orientation = piece_orientation_data_list[po]
    sym_index = orientation.piece_orientation.sym_index
    piece = ALL_PIECES[orientation.piece_orientation.piece_index]
    name = f'{piece.name}/{sym_index}'

    drawing = orientation.get_ascii_drawing((x, y))
    drawing_lines = drawing.splitlines()

    if c > 0:
        print('')
    print(f'  // {c}: {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    mask_array_start_index = piece_orientation_data.mask_array_start_index
    height = piece_orientation_data.height
    width = piece_orientation_data.width

    corner_offset = (y-1, x-1)
    print(f'  {{{{{y}, {x}}}, {p}, {po}}}{end}')

print('};  // kPieceOrientationCornerData')
print('')

def index_to_location(i):
    r = (i + 3) // 8
    c = (i + r + 2 + (i<5)) % 8 - 3
    return r, c


def location_to_index(r, c):
    assert 0 <= r <= 4 and -3 <= c <= 4, (r, c)
    i = 7 * r + c + 1 - (r < 1)
    rr, cc = index_to_location(i)
    assert (r, c) == (rr, cc), (i, r, c, rr, cc)
    return i


kMiniBoardLookup = []


print('const _MiniBoardLookup kMiniBoardLookup[kMiniBoardLookupSize] = {')

for c, piece_orientation_corner_data in enumerate(piece_orientation_corner_data_list):
    cx = piece_orientation_corner_data.x
    cy = piece_orientation_corner_data.y
    po = piece_orientation_corner_data.piece_orientation_index
    orientation = piece_orientation_data_list[po]

    relative_xy_list = []
    valid = True
    for x, y in orientation.piece_orientation.coordinates:
        x += 1
        y += 1
        if (y, x) < (cy, cx):
            valid = False
            break
        relative_xy_list.append((x - cx, y - cy))

    if not valid:
        continue

    mask = 0
    for x, y in relative_xy_list:
        i = location_to_index(y, x)
        mask |= 1 << i

    kMiniBoardLookup.append((mask, c))


kMiniBoardLookup.sort()


for i, (mask, c) in enumerate(kMiniBoardLookup):
    piece_orientation_corner_data = piece_orientation_corner_data_list[c]
    cx = piece_orientation_corner_data.x
    cy = piece_orientation_corner_data.y
    po = piece_orientation_corner_data.piece_orientation_index
    orientation = piece_orientation_data_list[po]

    sym_index = orientation.piece_orientation.sym_index
    piece = ALL_PIECES[orientation.piece_orientation.piece_index]
    name = f'{piece.name}/{sym_index}'
    drawing = orientation.get_ascii_drawing((cx, cy), True)
    drawing_lines = drawing.splitlines()

    print(f'  // {c}: {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    end = ',\n' if i < len(kMiniBoardLookup) - 1 else ''
    print(f'  {{{hex(mask)}, {c}}}{end}')

print('};  // kMiniBoardLookup', len(kMiniBoardLookup))
print('')

print('const uint8_t kPieceOrientationRowMasks[kNumPieceOrientationRowMasks] = {')

O = len(piece_orientation_data_list) - 1
for o, data in enumerate(piece_orientation_data_list):
    orientation = data.piece_orientation
    sym_index = orientation.sym_index
    piece = ALL_PIECES[orientation.piece_index]
    name = f'{piece.name}/{sym_index}'
    end = ',\n' if o < len(piece_orientation_data_list) - 1 else ''

    drawing = data.get_ascii_drawing()
    drawing_lines = drawing.splitlines()

    M = len(data.diagonal_masks) - 1

    if o > 0:
        print('')
    print(f'  // {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    comma = '' if o == O else ','

    main_str = ', '.join(f'0b{mask:08b}' for mask in data.row_masks) + ','
    adjacent_str = ', '.join(f'0b{mask:08b}' for mask in data.adjacency_masks) + ','
    diagonal_str = ', '.join(f'0b{mask:08b}' for mask in data.diagonal_masks) + comma

    k = max(len(main_str), len(adjacent_str), len(diagonal_str))

    main_blank = ' ' * (k - len(main_str))
    adjacent_blank = ' ' * (k - len(adjacent_str))
    diagonal_blank = ' ' * (k - len(diagonal_str))

    print(f'  {main_str}{main_blank}  // main')
    print(f'  {adjacent_str}{adjacent_blank}  // adjacent')
    print(f'  {diagonal_str}{diagonal_blank}  // diagonal')

print('};  // kPieceOrientationRowMasks')
print('')

print(
    'const piece_orientation_corner_index_t kCornerConstraintArray[kCornerConstraintArraySize] = {')

ccStrs = [
    'ccNone',
    'ccN',
    'ccE',
    'ccS',
    'ccW',
    'ccNE',
    'ccSE',
    'ccSW',
    'ccNW',
    ]

for p, piece_data in enumerate(piece_data_list):
    piece = piece_data.piece
    name = piece.name
    end = ',' if p < len(piece_data_list) - 1 else ''
    canonical_orientation = piece.orientations[0]

    drawing = canonical_orientation.get_ascii_drawing()
    drawing_lines = drawing.splitlines()[:-1]

    if p > 0:
        print('')
    print(f'  // {name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    last_p = p == len(piece_data_list) - 1
    lines = []
    comments = []
    for c in range(9):
        indices = piece_data.constraints_dict.get(c, [])
        if not indices:
            continue
        index_str = ', '.join(map(str, indices))
        lines.append(f'  {index_str}')
        comments.append(f'  // {ccStrs[c]}')

    n = len(lines)
    for i in range(n):
        line = lines[i]
        comment = comments[i]
        last_i = i == n - 1
        comma = '' if last_i and last_p else ','
        print(f'{line}{comma}{comment}')

print('};  // kCornerConstraintArray')
print('')

print('}  // namespace tables')
print('}  // namespace blokus')
