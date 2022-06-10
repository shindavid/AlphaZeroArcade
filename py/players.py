import collections
import sys

from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from game import ColorIndex, GameState, Move, MoveMask, NUM_COLORS, NUM_MOVES_BOUND, TuiGameManager, \
    coordinates_to_mask, mask_to_coordinates, is_subset_of, BOARD_SIZE, to_move_index, BoardLocation
from pieces import ALL_PIECES, Piece


class Player:
    def __init__(self, identifier):
        self.identifier = identifier
        self.color_index = None

    def receive_color_assignment(self, c: ColorIndex):
        self.color_index = c

    def __eq__(self, other):
        return type(self)==type(other) and self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __str__(self):
        return f'{type(self).__name__}({self.identifier})'

    @abstractmethod
    def get_move(self, state: GameState) -> Move:
        pass


def get_legal_moves(state: GameState, c: ColorIndex, piece: Piece) -> MoveMask:
    mask = np.zeros(NUM_MOVES_BOUND, dtype=bool)
    permissible_mask = state.permissible_matrix[c]
    required_coordinates = mask_to_coordinates(state.required_matrix[c])
    for orientation in piece.orientations:
        for oi in range(piece.size):
            o_xy = orientation.coordinates[oi].reshape((1, 2))
            for zx, zy in required_coordinates:
                # test an orientation of the piece with (ox, oy) == (zx, zy)
                z_xy = np.array([zx, zy], dtype=int).reshape((1, 2))
                shift_xy = z_xy - o_xy
                move_coordinates = orientation.coordinates + shift_xy
                if np.min(move_coordinates) < 0 or np.max(move_coordinates) >= BOARD_SIZE:
                    continue
                move_mask = coordinates_to_mask(move_coordinates)
                if is_subset_of(move_mask, permissible_mask):
                    mask[to_move_index(orientation.index, tuple(shift_xy.reshape((-1,))))] = 1

    return mask


class BasicPlayer0(Player):
    """
    Chooses uniformly at random among all legal moves.
    """
    def get_move(self, state: GameState) -> Move:
        c = self.color_index
        mask = np.zeros(NUM_MOVES_BOUND, dtype=bool)

        for piece_index in np.where(state.available_pieces[c])[0]:
            piece = ALL_PIECES[piece_index]
            mask |= get_legal_moves(state, c, piece)

        legal_moves = np.where(mask)[0]
        if not len(legal_moves):
            return Move.get_pass()
        move_index = np.random.choice(legal_moves)
        move = Move.from_index(move_index)
        return move


class BasicPlayer1(Player):
    """
    Like BasicPlayer0, but only considers moves that use the largest pieces.
    """
    def get_move(self, state: GameState) -> Move:
        c = self.color_index

        piece_map = collections.defaultdict(list)
        for piece_index in np.where(state.available_pieces[c])[0]:
            piece = ALL_PIECES[piece_index]
            piece_map[piece.size].append(piece)

        for size in reversed(sorted(piece_map)):
            mask = np.zeros(NUM_MOVES_BOUND, dtype=bool)
            for piece in piece_map[size]:
                mask |= get_legal_moves(state, c, piece)

            legal_moves = np.where(mask)[0]
            if len(legal_moves):
                move_index = np.random.choice(legal_moves)
                move = Move.from_index(move_index)
                return move

        return Move.get_pass()


def within_reach(state: GameState, c: ColorIndex, loc: BoardLocation) -> bool:
    """
    Returns true iff color c can occupy loc with their next move.
    """
    x, y = loc
    permissible_mask = state.permissible_matrix[c]
    required_coordinates = mask_to_coordinates(state.required_matrix[c])
    for zx, zy in required_coordinates:
        l1_distance = abs(x-zx) + abs(y-zy)
        if l1_distance > 4:
            continue
        for p in np.where(state.available_pieces[c])[0]:
            piece = ALL_PIECES[p]
            for orientation in piece.orientations:
                for oi in range(piece.size):
                    o_xy = orientation.coordinates[oi].reshape((1, 2))

                    # test an orientation of the piece with (ox, oy) == (zx, zy)
                    z_xy = np.array([zx, zy], dtype=int).reshape((1, 2))
                    shift_xy = z_xy - o_xy
                    move_coordinates = orientation.coordinates + shift_xy
                    if np.min(move_coordinates) < 0 or np.max(move_coordinates) >= BOARD_SIZE:
                        continue
                    move_mask = coordinates_to_mask(move_coordinates)
                    if is_subset_of(move_mask, permissible_mask):
                        return True

    return False


class BasicPlayer2(Player):
    """
    For each unoccupied square, s, and for each color, c, let A[s,c] equal 1 if c can occupy s with its next move, and
    0 else.

    Scores a given board state for color c by summing 4*A[s,c] - sum_{c' != c} A[s,c'] over all unoccupied squares s.

    Among all maximal-size moves, chooses one that results in maximal score.
    """
    def get_move(self, state: GameState) -> Move:
        c = self.color_index

        state2 = GameState()

        piece_map = collections.defaultdict(list)
        for piece_index in np.where(state.available_pieces[c])[0]:
            piece = ALL_PIECES[piece_index]
            piece_map[piece.size].append(piece)

        for size in reversed(sorted(piece_map)):
            max_score = -1
            candidate_moves = []
            for piece in piece_map[size]:
                mask = get_legal_moves(state, c, piece)

                for m in np.where(mask)[0]:
                    state2.copy_from(state)
                    move = Move.from_index(m)
                    state2.apply_move(c, move)
                    score = self.compute_score(state2)
                    if score > max_score:
                        max_score = score
                        candidate_moves = [move]
                    elif score == max_score:
                        candidate_moves.append(move)

            if candidate_moves:
                return np.random.choice(candidate_moves)
        return Move.get_pass()

    def compute_score(self, state: GameState) -> int:
        c = self.color_index

        opponent_c = [c2 for c2 in range(NUM_COLORS) if c != c2]
        score = 0
        for loc in zip(*np.where(state.permissible_matrix[c])):
            score += 4 * within_reach(state, c, loc)
            score -= sum([within_reach(state, c2, loc) for c2 in opponent_c])
        return score


def main():
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        print(f'Using random seed {seed}')
        np.random.seed(seed)

    player_classes = [BasicPlayer2, BasicPlayer1, BasicPlayer1, BasicPlayer1]
    players = [cls(i) for cls, i in zip(player_classes, range(NUM_COLORS))]
    manager = TuiGameManager(players)
    manager.run()

    num_games = 32
    score_counts = collections.defaultdict(float)

    print('')
    print(f'Simulating {num_games} games...')
    for _ in tqdm(range(num_games)):
        manager = TuiGameManager(players)
        winners = manager.run(silent=True)

        for w in winners:
            score_counts[w] += 1.0 / len(winners)

    print(f'Win counts after {num_games} games:')
    for player in players:
        count = score_counts[player]
        print(f'{player}: {count}')


if __name__ == '__main__':
    main()
