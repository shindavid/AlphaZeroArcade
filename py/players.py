import sys

import numpy as np

from game import ColorIndex, GameState, Move, NUM_COLORS, TuiGameManager


class RandomPlayer:
    def __init__(self, c: ColorIndex):
        self.c = c

    def get_move(self, state: GameState) -> Move:
        mask = state.get_legal_move_mask(self.c)
        legal_moves = np.where(mask)[0]
        move_index = np.random.choice(legal_moves)
        move = Move.from_index(move_index)
        return move


def main():
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        print(f'Using random seed {seed}')
        np.random.seed(seed)
    players = [RandomPlayer(c) for c in range(NUM_COLORS)]
    manager = TuiGameManager(players)
    manager.run()


if __name__ == '__main__':
    main()
