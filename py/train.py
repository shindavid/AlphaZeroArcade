import abc

from tqdm import tqdm

from blokus.game import GameState, Move
import copy
import numpy as np
from typing import List, Optional

from py.blokus.players import get_legal_moves

ActionIndex = int
ActionMask = np.ndarray
PlayerIndex = int


class AbstractAction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getIndex(self) -> ActionIndex:
        pass


class AbstractNeuralNetwork(metaclass=abc.ABCMeta):
    pass


class AbstractGameState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getNextState(self, player_index: PlayerIndex, action: AbstractAction) -> 'AbstractGameState':
        pass

    @abc.abstractmethod
    def getValidActions(self, player_index: PlayerIndex) -> ActionMask:
        pass

    @abc.abstractmethod
    def getGameEnded(self, player_index: PlayerIndex) -> bool:
        pass


class BlokusAction(AbstractAction):
    def toMove(self) -> Move:
        pass


class BlokusGameState(AbstractGameState):
    def __init__(self, num_players: int, state: Optional[GameState]=None):
        assert num_players in (2, 4)
        self._num_players = num_players
        self._state = GameState() if state is None else state

    def getNextState(self, player_index: PlayerIndex, action: BlokusAction) -> 'BlokusGameState':
        state = copy.deepcopy(self._state)
        state.apply_move(state.get_current_color_index(), action.toMove())
        return BlokusGameState(self._num_players, state)

    def getValidActions(self, player_index: PlayerIndex) -> ActionMask:
        return get_legal_moves(self._state, self._state.get_current_color_index())
        pass

    def getGameEnded(self, player_index: PlayerIndex) -> bool:
        return self._state.getGameEnded()


class BlokusNeuralNetwork:
    pass


class MCTS:
    def __init__(self):
        # TODO
        pass


class Coach:
    def __init__(self, state: AbstractGameState, nnet: AbstractNeuralNetwork):
        self.args = {
            'numIters': 1024,
            'numEps': 64
        }

        self.state = state
        self.nnet = nnet

    def learn(self):
        for i in range(1024):
            print(f'Coach.learn() i={i}')
            train_examples = []  # deque?

            for _ in tqdm(range(self.args['numEps']), desc="Self Play"):
                self.mcts = MCTS(self.state, self.nnet, self.args)  # reset search tree
                train_examples += self.executeEpisode()


def main():
    state = BlokusGameState(num_players=2)
    nnet = BlokusNeuralNetwork()
    coach = Coach(state, nnet)
    coach.learn()


if __name__ == '__main__':
    main()
