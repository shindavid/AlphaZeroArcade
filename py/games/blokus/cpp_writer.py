from games.blokus.pieces import ALL_PIECES, ALL_PIECE_ORIENTATIONS

from collections import defaultdict


def extract_column_masks(coordinates):
    masks = defaultdict(int)

    for x, y in coordinates:
        masks[x] |= 1 << y

    columns = []
    for m in sorted(masks):
        columns.append(bin(masks[m]))
    return columns


print('// Auto-generated by py/games/blokus/cpp_writer.py')

header = """
#include <games/blokus/Constants.hpp>
#include <games/blokus/Pieces.hpp>

namespace blokus {
"""
print(header)
print('const Piece kPieces[kNumPieces] = {')
for p, piece in enumerate(ALL_PIECES):
    name = piece.name
    sym_subgroup = piece.sym_subgroup
    end = ',' if p < len(ALL_PIECES) - 1 else ''
    canonical_orientation = piece.orientations[0]

    drawing = canonical_orientation.get_ascii_drawing()
    drawing_lines = drawing.splitlines()[:-1]

    print('')
    print(f'  // p{name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    o_index = canonical_orientation.index
    coords_str = ', '.join(extract_column_masks(canonical_orientation.coordinates))
    print(f'  Piece("{name}", p{name}, {o_index}, g{sym_subgroup}, {coords_str}){end}')

print('};  // kPieces')
print('')
print('const PieceOrientation kPieceOrientations[kNumPieceOrientations] = {')
for o, orientation in enumerate(ALL_PIECE_ORIENTATIONS):
    sym_index = orientation.sym_index
    piece = ALL_PIECES[orientation.piece_index]
    name = f'{piece.name}/{sym_index}'
    end = ',' if o < len(ALL_PIECE_ORIENTATIONS) - 1 else ''

    drawing = orientation.get_ascii_drawing()
    drawing_lines = drawing.splitlines()[:-1]

    print('')
    print(f'  // p{name}')
    print('  //')
    for line in drawing_lines:
        print(f'  // {line}')

    coords_str = ', '.join(extract_column_masks(orientation.coordinates))
    print(f'  PieceOrientation({o}, p{piece.name}, {sym_index}, {coords_str}){end}')

print('};  // kPieceOrientations')
print('')

print('}  // namespace blokus')
print('')