from blokus.game import GameState, Move
from blokus.pieces import ALL_PIECES

import abc
import collections
import copy
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional

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
    def getNextState(self, action_index: ActionIndex) -> 'AbstractGameState':
        pass

    @abc.abstractmethod
    def getValidActions(self) -> ActionMask:
        pass

    @abc.abstractmethod
    def getGameEnded(self) -> bool:
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

    def getGameEnded(self) -> bool:
        return self._state.getGameEnded()

    def __hash__(self) -> int:
        return hash(self._state)

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False        
        return self._state == other._state


class BlokusNeuralNetwork(AbstractNeuralNetwork):
    pass


class MCTSNodeStats:
    def __init__(self):
        self.count = 0
        self.value_sum = 0
        self.policy_prior = 0.0


class MCTS:
    def __init__(self, state: AbstractGameState,
                 network: AbstractNeuralNetwork,
                 args: Dict[str, Any]):
        self.game_state_stats = collections.defaultdict(MCTSNodeStats)
        self.forward_adj_list = collections.defaultdict(list)
        self.parents = {}

        self.game_state_stats[(0, state)].count += 1

        valid_action_mask = state.getValidActions()
        for valid_action_index in np.where(valid_action_mask)[0]:
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
