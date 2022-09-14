import copy
import sys
from typing import Optional

from game import GameState, Move


sys.path.append('..')
from train import AbstractGameState, ActionIndex, ActionMask, ValueProbDistr, AbstractNeuralNetwork


class BlokusGameState(AbstractGameState):
    def __init__(self, num_players: int, state: Optional[GameState]=None):
        assert num_players in (2, 4)
        self._num_players = num_players
        self._state = GameState() if state is None else state

    def getNextState(self, action_index: ActionIndex) -> 'BlokusGameState':
        state = copy.deepcopy(self._state)
        state.apply_move(Move.from_index(action_index))
        return BlokusGameState(self._num_players, state)

    def getValidActions(self) -> ActionMask:
        return self._state.get_legal_moves()

    def getGameResult(self) -> Optional[ValueProbDistr]:
        raise Exception('TODO')

    def __hash__(self) -> int:
        return hash(self._state)

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        return self._state == other._state


class BlokusNeuralNetwork(AbstractNeuralNetwork):
    pass


def main():
    state = BlokusGameState(num_players=2)
    nnet = BlokusNeuralNetwork()
    # TODO


if __name__ == '__main__':
    main()
