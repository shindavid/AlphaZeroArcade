import numpy as np


PlayerIndex = int
BoardMask = np.ndarray  # 20x20 bool


class Move:
    # represents (piece, orientation, location) tuple
    def to_mask(self) -> BoardMask:
        # deserves care to make sure this is efficiently computable
        # 91 x 20 x 20 = 36,400
        # brute force memory usage: (91 x 20 x 20) -> (20 x 20)  [400 bits * 36,400 = 14,560,000 = 14MB]
        # so some cheap calc instead would be better.
        pass


NUM_MOVES_BOUND = 91 * 20 * 20

MoveMask = np.ndarray  # NUM_MOVES_BOUND bool


def is_subset_of(sub_mask: BoardMask, super_mask: BoardMask) -> bool:
    pass


class GameState:
    def __init__(self):
        self._occupied_spaces = np.ndarray((4, 20, 20), dtype=bool)

    def is_piece_available(self, p: PlayerIndex, m: Move) -> bool:
        pass

    def get_occupied(self, p: PlayerIndex) -> BoardMask:
        pass

    def get_adjacent_neighbors(self, p: PlayerIndex) -> BoardMask:
        pass

    def get_diagonal_neighbors(self, p: PlayerIndex) -> BoardMask:
        pass

    def get_unoccupied(self) -> BoardMask:
        pass

    def get_y_mask(self, p: PlayerIndex) -> BoardMask:
        pass

    def get_z_mask(self, p: PlayerIndex) -> BoardMask:
        pass

    def is_legal(self, p: PlayerIndex, move: Move) -> bool:
        # M is legal iff Z \subset M \subset Y
        # TODO: validate that piece has not been used yet.
        if not self.is_piece_available(p, move):
            return False
        m = move.to_mask()
        z = self.get_z_mask(p)
        return is_subset_of(z, m) and is_subset_of(m, self.get_y_mask(p))

    def get_legal_move_mask(self, p: PlayerIndex) -> MoveMask:
        pass

        # want:
        # for player p in (0,1,2,3)...
        #  - want legal spaces I can occupy (i.e. spaces that are not occupied & not adjacent to my current squares)
        #  - also want spaces which I MUST occupy at least one of (i.e., among above spaces, those which are diagonally adjacent to my current)
        #
        # xx....ooo
        # xxx......
        # ...xxx...
        # ...x.....   X = set of all squares I currently occupy
        # ...x.....
        # .........
        #
        # ...yyy...
        # ......yyy
        # .......yy
        # yy....yyy   Y = set of all unoccupied squares - {adjacent neighbors of X}
        # yy...yyyy
        # yyy.yyyyy
        #
        # ...z.....
        # ......z..
        # .........   Z = Y intersected with {diagonal neighbors of X}
        # ......z..
        # .........
        # ..z.z....

        # Let M be the subset of the board your next piece covers
        # M is legal iff Z \subset M \subset Y
