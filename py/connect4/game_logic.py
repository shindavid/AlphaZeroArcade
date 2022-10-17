import os
import random
import sys
from typing import List, Hashable

import numpy as np
import torch
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from interface import AbstractGameState, ActionIndex, ActionMask, GameResult

NUM_COLUMNS = 7
NUM_ROWS = 6
MAX_MOVES_PER_GAME = NUM_ROWS * NUM_COLUMNS
Color = int


COLORS = ['R', 'Y']
NUM_COLORS = len(COLORS)
CIRCLE_CHAR = chr(9679)
LEFT_TACK = chr(8867)
RIGHT_TACK = chr(8866)
PRETTY_COLORS = [colored(CIRCLE_CHAR, c) for c in ('red', 'yellow')]


class C4GameState(AbstractGameState):
    RED = 0  # first player
    YELLOW = 1

    def __init__(self, piece_mask=None, current_player=None):
        self.piece_mask = np.zeros((2, NUM_COLUMNS, NUM_ROWS), dtype=bool) if piece_mask is None else piece_mask
        self.current_player = C4GameState.RED if current_player is None else current_player

    def clone(self) -> 'C4GameState':
        return C4GameState(np.copy(self.piece_mask), self.current_player)

    def get_signature(self) -> Hashable:
        return self.piece_mask.data.tobytes(), self.current_player

    @staticmethod
    def get_num_global_actions() -> int:
        return NUM_COLUMNS

    def debug_dump(self, file_handle):
        file_handle.write(self.to_ascii_drawing(pretty_print=False))

    def compact_repr(self) -> str:
        return self.get_board_str()

    def get_num_moves_played(self) -> int:
        return int(np.sum(self.piece_mask))

    def get_valid_moves(self) -> List[int]:
        cur_heights = np.sum(np.sum(self.piece_mask, axis=0), axis=1)
        assert cur_heights.shape == (NUM_COLUMNS, )
        return [c+1 for c, h in enumerate(cur_heights) if h < NUM_ROWS]

    def get_valid_actions(self) -> ActionMask:
        actions = np.array(self.get_valid_moves())
        mask = torch.zeros(C4GameState.get_num_global_actions(), dtype=bool)
        mask[actions-1] = 1
        return mask

    def get_current_player(self) -> Color:
        return self.current_player

    def get_mask(self, color: Color) -> np.ndarray:
        return self.piece_mask[color]

    def undo_move(self, column: int):
        """
        Assumes that self.apply_move(column) was just invoked, and undoes that call.
        """
        assert 1 <= column <= NUM_COLUMNS

        cur_height = np.sum(self.piece_mask[:, column-1])
        assert cur_height > 0
        self.piece_mask[:, column-1, cur_height-1] = 0
        self.current_player = 1 - self.current_player

    def apply_move(self, action_index: ActionIndex) -> GameResult:
        winners = self._add_piece(action_index + 1)
        if winners:
            arr = np.zeros(2)
            arr[winners] = 1.0 / len(winners)
            return arr
        return None

    def _add_piece(self, column: int, announce: bool = False) -> List[Color]:
        """
        column is 1-indexed

        Returns the winning colors if the game has ended.
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

        winners = self._compute_winners(column, cur_height, move_color)

        if announce and winners:
            cur_color = 'R' if self.current_player == C4GameState.RED else 'Y'
            print(f'** {cur_color} playing in column {column}')
            if len(winners) == 2:
                print('Tied!')
            else:
                print(f'Winner! {cur_color}')
        return winners

    def _compute_winners(self, column, cur_height, move_color):
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
                    return [move_color]
        if self.get_num_moves_played() == MAX_MOVES_PER_GAME:
            return [C4GameState.RED, C4GameState.YELLOW]
        return []

    def get_board_str(self) -> str:
        tokens = ['.' for _ in range(NUM_ROWS * NUM_COLUMNS)]
        for c, color in enumerate(COLORS):
            for k in np.where(self.piece_mask[c].reshape((-1,)))[0]:
                tokens[k] = color
        return ''.join(tokens)

    def to_ascii_drawing(self, pretty_print=True, newline=True, add_legend=False, player_names=('1', '2'),
                         highlight_column=None) -> str:
        colors = PRETTY_COLORS if pretty_print else COLORS
        char_matrix = [[' ' for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]
        blink_x = -1
        blink_y = -1
        for (c, color) in enumerate(colors):
            for x, y in zip(*np.where(self.piece_mask[c])):
                char_matrix[y][x] = color
                if x+1 == highlight_column:
                    blink_x = x
                    blink_y = max(y, blink_y)

        if blink_x >= 0:
            char_matrix[blink_y][blink_x] = colored(char_matrix[blink_y][blink_x], attrs=['blink'])

        out_lines = []
        for char_list in reversed(char_matrix):
            body = '|'.join(char_list)
            out_lines.append('|' + body + '|')

        if add_legend:
            out_lines.append('|1|2|3|4|5|6|7|')
            out_lines.append(f'{colors[0]}: {player_names[0]}')
            out_lines.append(f'{colors[1]}: {player_names[1]}')
        if newline:
            out_lines.append('')
        return '\n'.join(out_lines)

    def __hash__(self):
        return hash(tuple(self.piece_mask.data))

    def __eq__(self, other: 'C4GameState'):
        return all(np.all(m1 == m2) for m1, m2 in (self.piece_mask, other.piece_mask))


if __name__ == '__main__':
    for _ in range(1000):
        game = C4GameState()
        while True:
            moves = game.get_valid_moves()
            assert moves
            results = game._add_piece(random.choice(moves), announce=False)
            if results:
                break

    print('Played all games!')
