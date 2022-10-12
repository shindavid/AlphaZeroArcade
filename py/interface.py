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
GameResult = Optional[ValueProbDistr]
Shape = Tuple[int, ...]


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def evaluate(self, vec: NeuralNetworkInput) -> Tuple[GlobalPolicyLogitDistr, ValueLogitDistr]:
        pass


class AbstractGameState(ABC):
    @staticmethod
    @abstractmethod
    def get_num_global_actions() -> int:
        """
        Returns the number of global actions.
        """
        pass

    @abstractmethod
    def debug_dump(self, file_handle):
        """
        For debugging.
        """
        pass

    @abstractmethod
    def compact_repr(self) -> str:
        """
        For debugging.
        """
        pass

    @abstractmethod
    def get_current_player(self) -> PlayerIndex:
        """
        Returns the player that would correspond to the next apply_move() call.
        """
        pass

    @abstractmethod
    def apply_move(self, action_index: ActionIndex) -> GameResult:
        """
        Apply a move, modifying the game state.

        Returns None if game is not over. Else, returns a vector of non-negative win/loss values summing to 1.
        These values can be fractional in the event of a tie.
        """
        pass

    @abstractmethod
    def get_valid_actions(self) -> ActionMask:
        """
        Returns a mask indicating which moves are legal at the current state.
        """
        pass

    @abstractmethod
    def get_signature(self) -> Hashable:
        """
        Returns an object that will serve as the dict-key for a neural network evaluation cache.

        This can be simply :self:
        """
        pass


class AbstractGameTensorizor(ABC):
    @abstractmethod
    def vectorize(self, state: AbstractGameState) -> NeuralNetworkInput:
        """
        Returns a tensor that will be fed into the neural network.
        """
        pass

    @abstractmethod
    def receive_state_change(self,  state: AbstractGameState, action_index: ActionIndex):
        """
        Notify the tensorizor of a new game state. This can be useful if move history
        information is part of the vectorized representation of the state.
        """
        pass

    @staticmethod
    @abstractmethod
    def supports_undo() -> bool:
        """
        Returns whether the class implements an undo_last_move() method. If it does, the MCTS implementation will use
        it. Otherwise, the class must support a clone() method.
        """
        pass

    def undo(self, state: AbstractGameState):
        """
        If supports_undo() returns True, then this method must be implemented.

        Calling the method should effectively undo both the last state.apply_move() call and the last
        self.receive_state_change() call.
        """
        raise NotImplemented()

    def clone(self, state: AbstractGameState) -> Tuple['AbstractGameTensorizor', AbstractGameState]:
        """
        If supports_undo() returns False, then this method must be implemented.

        Returns a deep copy of self, and a deep copy of state.
        """
        raise NotImplemented()
