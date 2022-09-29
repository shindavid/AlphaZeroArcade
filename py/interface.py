from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Hashable


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
    def supports_undo() -> bool:
        """
        Returns whether the class implements an undo_last_move() method. If it does, the MCTS implementation will use it.
        Otherwise, the class must support a clone() method.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_num_global_actions() -> int:
        """
        Returns the number of global actions.
        """
        pass

    def debug_dump(self, file_handle):
        """
        For debugging.
        """
        pass

    @abstractmethod
    def get_signature(self) -> Hashable:
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
    def get_current_player(self) -> PlayerIndex:
        """
        Returns the player that would correspond to the next apply_move() call.
        """
        pass

    @abstractmethod
    def apply_move(self, action_index: ActionIndex):
        """
        Apply a move, modifying the game state.
        """
        pass

    def undo_last_move(self):
        """
        If supports_undo() returns True, then this method must be implemented. Calling the method should effectively
        undo the last apply_move() call.
        """
        raise NotImplementedError()

    def clone(self):
        """
        If supports_undo() returns False, then this method must be implemented. Returns a deep copy of self.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_valid_actions(self) -> ActionMask:
        """
        Returns a mask indicating which moves are legal at the current state.
        """
        pass

    @abstractmethod
    def get_game_result(self) -> Optional[ValueProbDistr]:
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
