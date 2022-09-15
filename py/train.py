from torch import Tensor

import abc
import numpy as np
from typing import Optional, Tuple


ActionIndex = int
ActionMask = np.ndarray  # bool
PlayerIndex = int
NeuralNetworkInput = Tensor
GlobalPolicyProbDistr = Tensor  # size=# global actions, prob terms
GlobalPolicyLogitDistr = Tensor  # size=# global actions, logit terms
LocalPolicyProbDistr = Tensor  # size=# locally legal actions, prob terms
LocalPolicyLogitDistr = Tensor  # size=# locally legal actions, logit terms
ValueProbDistr = Tensor  # size=# players, prob terms
ValueLogitDistr = Tensor  # size=# players, logit terms


class AbstractNeuralNetwork(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, vec: NeuralNetworkInput) -> Tuple[GlobalPolicyLogitDistr, ValueLogitDistr]:
        pass


class AbstractGameState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getCurrentPlayer(self) -> PlayerIndex:
        pass

    @abc.abstractmethod
    def getNextState(self, action_index: ActionIndex) -> 'AbstractGameState':
        pass

    @abc.abstractmethod
    def getValidActions(self) -> ActionMask:
        pass

    @abc.abstractmethod
    def getGameResult(self) -> Optional[ValueProbDistr]:
        """
        Returns None if game is not over. Else, returns a vector of win/loss values (whose sum
        should typically be 1).
        """
        pass

    @abc.abstractmethod
    def vectorize(self) -> NeuralNetworkInput:
        pass

    @abc.abstractmethod
    def getNumGlobalActions(self) -> int:
        pass
