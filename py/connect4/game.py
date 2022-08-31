import random
import time
from typing import Optional, List

import numpy as np
from termcolor import colored


NUM_COLUMNS = 7
NUM_ROWS = 6
Color = int


COLORS = ['R', 'Y']
NUM_COLORS = len(COLORS)
CIRCLE_CHAR = chr(9679)
PRETTY_COLORS = [colored(CIRCLE_CHAR, c) for c in ('red', 'yellow')]


T1 = 0
T2 = 0
N = 0


def compute_segment_masks():
    """
    Returns a list SEGMENT_MASKS with the property that...

    SEGMENT_MASKS[col][row] returns a list of (c_tups, r_tups),
    with the property that for every 4-segment {(c0, r0), (c1, r1), (c2, r2), (c3, r3)}
    on the board, SEGMENT_MASKS[ci][ri] contains the 2-tuple
    ((c0, c1, c2, c3), (r0, r1, r2, r3)).

    This structure is useful for computing winner information.
    """
    output = []
    for _ in range(NUM_COLUMNS):
        output.append([[] for _ in range(NUM_ROWS)])

    for c in range(NUM_COLUMNS):
        for r in range(NUM_ROWS):
            for (dc, dr) in ((0, 1), (1, 0), (1, 1), (1, -1)):
                c2 = c + 3*dc
                r2 = r + 3*dr
                if not (0 <= c2 < NUM_COLUMNS):
                    continue
                if not (0 <= r2 < NUM_ROWS):
                    continue

                c_tup = tuple(c+k*dc for k in range(4))
                r_tup = tuple(r+k*dr for k in range(4))

                for k in range(4):
                    cc = c + k * dc
                    rr = r + k * dr
                    output[cc][rr].append((c_tup, r_tup))

    return output


SEGMENT_MASKS = compute_segment_masks()


class Game:
    RED = 0  # first player
    YELLOW = 1

    def __init__(self):
        self.piece_mask = np.zeros((2, NUM_COLUMNS, NUM_ROWS), dtype=bool)
        self.current_player = Game.RED

    def get_valid_moves(self) -> List[int]:
        cur_heights = np.sum(np.sum(self.piece_mask, axis=0), axis=1)
        assert cur_heights.shape == (NUM_COLUMNS, )
        return [c+1 for c, h in enumerate(cur_heights) if h + 1 < NUM_ROWS]

    def apply_move(self, column: int, announce: bool=False) -> Optional[Color]:
        """
        column is 1-indexed

        Returns the winning color if the game has ended. Else returns None.
        """
        assert 1 <= column <= NUM_COLUMNS

        cur_height = np.sum(self.piece_mask[:, column-1])
        key = (self.current_player, column-1, cur_height)
        key2 = (1-self.current_player, column-1, cur_height)
        assert self.piece_mask[key] == 0
        assert self.piece_mask[key2] == 0
        self.piece_mask[key] = 1

        move_color = self.current_player
        self.current_player = 1 - self.current_player

        global T1, T2, N

        t1 = time.time()
        w1 = self.compute_winner1(column, cur_height, move_color)
        t2 = time.time()
        w2 = self.compute_winner2(column, cur_height, move_color)
        t3 = time.time()

        T1 += t2 - t1
        T2 += t3 - t2
        N += 1

        assert w1 == w2, (w1, w2, self.to_ascii_drawing())
        if announce and w2 is not None:
            cur_color = 'R' if self.current_player == Game.RED else 'Y'
            print(f'** {cur_color} playing in column {column}')
            print(f'Winner! {cur_color}')
        return w2

    def compute_winner1(self, column, cur_height, move_color):
        mask = self.piece_mask[move_color]
        dir_tuple_set = (
            ((-1, 0), (+1, 0)),  # horizontal
            ((0, -1),),  # vertical
            ((-1, -1), (+1, +1)),  # diagonal 1
            ((-1, +1), (+1, -1)),  # diagonal 2
        )
        for dir_tuples in dir_tuple_set:
            count = 1
            for (dc, dr) in dir_tuples:
                c = column - 1
                r = cur_height
                while count < 4:
                    c += dc
                    r += dr
                    if 0 <= c < NUM_COLUMNS:
                        if 0 <= r < NUM_ROWS:
                            if mask[(c, r)]:
                                count += 1
                                continue
                    break
                if count == 4:
                    return move_color
        return None

    def compute_winner2(self, column, cur_height, move_color):
        mask = self.piece_mask[move_color]
        seg = SEGMENT_MASKS[column-1][cur_height]
        if any(np.sum(mask[s]) == 4 for s in seg):
            return move_color
        return None

    def to_ascii_drawing(self, pretty_print=True, newline=True) -> str:
        colors = PRETTY_COLORS if pretty_print else COLORS
        empty_color = ' ' if pretty_print else '.'
        char_matrix = [[empty_color for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]
        for (c, color) in enumerate(colors):
            for x, y in zip(*np.where(self.piece_mask[c])):
                char_matrix[y][x] = color
        out_lines = list(reversed([''.join(char_list) for char_list in char_matrix]))
        if newline:
            out_lines.append('')
        return '\n'.join(out_lines)

    def __hash__(self):
        return hash(tuple(self.piece_mask.data))

    def __eq__(self, other: 'Game'):
        return all(np.all(m1 == m2) for m1, m2 in (self.piece_mask, other.piece_mask))


if __name__ == '__main__':
    for _ in range(1000):
        g = Game()
        while True:
            moves = g.get_valid_moves()
            if not moves:
                #print('Game is drawn!')
                break
            result = g.apply_move(random.choice(moves), announce=False)
            #print(g.to_ascii_drawing())
            if result is not None:
                break

    print('Played all games!')
    print('t1: %.fns' % (T1*1e9/N))
    print('t2: %.fns' % (T2*1e9/N))
