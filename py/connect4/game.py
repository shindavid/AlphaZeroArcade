from typing import Optional

import numpy as np


Color = int


class Game:
    RED = 0  # first player
    YELLOW = 1

    def __init__(self, num_columns: int=7, num_rows=6):
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.piece_mask = np.zeros((2, self.num_columns, self.num_rows), dtype=bool)
        self.current_player = Game.RED

    def apply_move(self, column: int) -> Optional[Color]:
        """
        column is 1-indexed

        Returns the winning color if the game has ended. Else returns None.
        """
        assert 1 <= column <= self.num_columns
        cur_height = np.sum(self.piece_mask[:, column-1])
        key = (self.current_player, column-1, cur_height)
        assert self.piece_mask[key] == 0
        self.piece_mask[key] = 1

        move_color = self.current_player
        mask = self.piece_mask[move_color]
        self.current_player = 1 - self.current_player

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
                    if 0 <= c < self.num_columns:
                        if 0 <= r < self.num_rows:
                            if mask[(c, r)]:
                                count += 1
                                continue
                    break
                if count == 4:
                    return move_color
        return None

    def __str__(self):
        raise Exception('TODO')

    def __hash__(self):
        return hash(tuple(self.piece_mask.data))

    def __eq__(self, other: 'Game'):
        return all(np.all(m1 == m2) for m1, m2 in (self.piece_mask, other.piece_mask))
