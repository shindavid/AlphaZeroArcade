from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Hashable


"""
Various type aliases.

Global* types are global in the sense that they correspond to the set of all possible moves in the game.
Local* types are local in the sense that they correspond only to the set of legal moves in a given state.

The Prob vs Logit identifiers in the type aliases refer to whether the values refer to probabilities or logit values.
Logit values can be converted to probabilities via softmax.

TODO: ActionIndex/ActionMask should probably be specialized further to Local/Global variants.
"""
ActionIndex = int
ActionMask = Tensor  # bool
PlayerIndex = int
NeuralNetworkInput = Tensor
GlobalPolicyProbDistr = Tensor  # size=# global actions, prob terms
GlobalPolicyLogitDistr = Tensor  # size=# global actions, logit terms
LocalPolicyProbDistr = Tensor  # size=# locally legal actions, prob terms
LocalPolicyLogitDistr = Tensor  # size=# locally legal actions, logit terms
ValueProbDistr = Tensor  # size=# players, prob terms
ValueLogitDistr = Tensor  # size=# players, logit terms


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def evaluate(self, vec: NeuralNetworkInput) -> Tuple[GlobalPolicyLogitDistr, ValueLogitDistr]:
        pass


class AbstractGameState(ABC):
    @staticmethod
    @abstractmethod
    def supportsUndo() -> bool:
        """
        Returns whether the class implements an undoLastMove() method. If it does, the MCTS implementation will use it.
        Otherwise, the class must support a clone() method.
        """
        pass

    @staticmethod
    @abstractmethod
    def getNumGlobalActions() -> int:
        """
        Returns the number of global actions.
        """
        pass

    def debugDump(self, file_handle):
        """
        For debugging.
        """
        pass

    @abstractmethod
    def getSignature(self) -> Hashable:
        """
        Returns an object that will serve as the dict-key for a neural network evaluation cache.

        This can be simply <self>, provided the class has appropriate __eq__/__hash__ implementations that effectively
        use the data structures from which the output of vectorize() are computed.

        However, that behavior can be suboptimal because sometimes you want two states with unequal vectorize() outputs
        to share a common signature. This is because the neural network input typically includes recent state history,
        and because different move orders can transpose to the same position. Technically, the neural network will not
        yield identical outputs for such transpositions, but game theoretically the evaluations should be identical.
        So a theoretically perfect neural network should yield identical outputs, justifying a shared cache entry.
        """
        pass

    @abstractmethod
    def getCurrentPlayer(self) -> PlayerIndex:
        """
        Returns the player that would correspond to the next applyMove() call.
        """
        pass

    @abstractmethod
    def applyMove(self, action_index: ActionIndex):
        """
        Apply a move, modifying the game state.
        """
        pass

    def undoLastMove(self):
        """
        If supportsUndo() returns True, then this method must be implemented. Calling the method should effectively
        undo the last applyMove() call.
        """
        raise NotImplementedError()

    def clone(self):
        """
        If supportsUndo() returns False, then this method must be implemented. Returns a deep copy of self.
        """
        raise NotImplementedError()

    @abstractmethod
    def getValidActions(self) -> ActionMask:
        """
        Returns a mask indicating which moves are legal at the current state.
        """
        pass

    @abstractmethod
    def getGameResult(self) -> Optional[ValueProbDistr]:
        """
        Returns None if game is not over. Else, returns a vector of non-negative win/loss values summing to 1.
        These values can be fractional in the event of a tie.
        """
        pass

    @abstractmethod
    def vectorize(self) -> NeuralNetworkInput:
        """
        Returns a tensor that will be fed into the neural network.
        """
        pass
