from blokus.game import GameState, Move

import abc
import copy
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

from py.mcts import MCTS

ActionIndex = int
ActionMask = np.ndarray  # bool
PlayerIndex = int
NeuralNetworkInput = np.ndarray
PolicyDistribution = np.ndarray  # float, sum=1, size=# actions
ValueDistribution = np.ndarray  # float, size=# players


class AbstractAction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getIndex(self) -> ActionIndex:
        pass


class AbstractNeuralNetwork(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, vec: NeuralNetworkInput) -> Tuple[PolicyDistribution, ValueDistribution]:
        pass


class AbstractGameState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getNextState(self, action_index: ActionIndex) -> 'AbstractGameState':
        pass

    @abc.abstractmethod
    def getValidActions(self) -> ActionMask:
        pass

    @abc.abstractmethod
    def getGameResult(self) -> Optional[ValueDistribution]:
        """
        Returns None if game is not over. Else, returns a vector of win/loss values (whose sum
        should typically be 1).
        """
        pass

    @abc.abstractmethod
    def vectorize(self) -> NeuralNetworkInput:
        pass


class BlokusAction(AbstractAction):
    def toMove(self) -> Move:
        pass


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

    def getGameResult(self) -> Optional[ValueDistribution]:
        raise Exception('TODO')

    def __hash__(self) -> int:
        return hash(self._state)

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False        
        return self._state == other._state


class BlokusNeuralNetwork(AbstractNeuralNetwork):
    pass


class Coach:
    def __init__(self, state: AbstractGameState, nnet: AbstractNeuralNetwork):
        self.args = {
            'treeSizeLimit': 1024,
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
