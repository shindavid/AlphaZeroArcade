from blokus.game import GameState, Move
from blokus.pieces import ALL_PIECES

import abc
import collections
import copy
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

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


class MCTSNodeStats:
    def __init__(self):
        self.count = 0
        self.value_sum = 0
        self.policy_prior = 0.0


Depth = int
MCTSNodeKey = Tuple[Depth, AbstractGameState]


class MCTS:
    """
    Constructed once per move of a game.
    """
    def __init__(self, state: AbstractGameState,
                 network: AbstractNeuralNetwork,
                 args: Dict[str, Any]):
        self.args = dict(args)

        self.network = network
        self.game_state_stats: Dict[MCTSNodeKey, MCTSNodeStats] = collections.defaultdict(MCTSNodeStats)
        self.fwd_adj_list: Dict[MCTSNodeKey, List[MCTSNodeKey]] = collections.defaultdict(list)
        self.rev_adj_list: Dict[MCTSNodeKey, MCTSNodeKey] = {}

        for _ in range(self.args['treeSizeLimit']):
            self.visit(0, state)

    def visit(self, depth: Depth, state: AbstractGameState):
        """
        [Katago paper]

        At node n, choose child c that maximizes:

        PUCT(c) = V(c) + c_{PUCT}*P(c)*sqrt(sum_{c'} N(c')) / (1 + N(c))

        where:
        - V(c): avg predicted utility of all nodes in c's subtree
        - P(c): policy prior of c from net
        - N(c): # playouts previously sent thru c
        - c_{PUCT}: 1.1

        KataGo adds noise to policy prior P(c) at the root:

        P(c) = 0.75*P_{raw}(c) + 0.25*nu

        where nu is a draw from a Dirichlet distribution on legal moves with param alpha = 0.03*19^2 / N(c),
        where N is the total number of legal moves. KataGo also applies a softmax temp at the root of 1.03

        ** Playout Cap Randomization **

        - with probability p ~= 0.25, we do a FULL SEARCH:
          - tree grows to size N
          - enable Dirichlet noise + explorative settings
          - export for nnet training
        - else, we do FAST SEARCH:
          - tree grows to size n < N
          - disable Dirichlet noise + explorative settings (maximizing strength)
          - do NOT export for nnet training

        [Rob] It is possible for multi-head nnets that predict y1 and y2 to use (x, y1) for training, without
        y2. In the KataGo context, it perhaps makes sense to utilize the value output for nnet training even
        for fast searches.
        """
        key = (depth, state)
        is_root = (depth == 0)
        use_noise = False  # TODO

        self.game_state_stats[key].count += 1
        game_result: Optional[ValueDistribution] = state.getGameResult()

        valid_action_mask = state.getValidActions()
        policy_distr, value_distr = self.network.evaluate(state.vectorize())
        if game_result is None:
            policy_distr *= valid_action_mask
            policy_sum = sum(policy_distr)
            assert policy_sum > 0.0, f'uh oh {policy_sum} {sum(valid_action_mask)}'
            policy_distr *= 1.0 / sum(policy_distr)

            policy_UCT_distr = policy_distr * self.c_PUCT *  # TODO

        for valid_action_index in np.where(valid_action_mask)[0]:
            # TODO
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
