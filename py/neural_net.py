"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import abc
import copy
import os
from typing import Dict, List, Optional

import torch
from torch import nn as nn

from util.torch_util import Shape

ENABLE_CUDA = True


class LearningTarget:
    """
    A LearningTarget corresponds to an output head of a NeuralNet. It bundles a number of useful pieces of information
    together:

    - The name of the head, to facilitate matching up with c++ generated data
    - The loss function to use for this head
    - The weight to assign to the loss produced by this head
    - Whether or how to mask rows of data
    - An accuracy measurement function
    """
    def __init__(self, name: str, loss_weight: float):
        self.name = name
        self.loss_weight = loss_weight

    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass

    def get_mask(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Assumes labels is a 2D tensor of shape (batch_size, num_labels). Returns a 1D tensor of shape (batch_size,)

        If no mask should be applied, return None. This is the default behavior; derived classes can override this.
        """
        return None

    @abc.abstractmethod
    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        pass


class PolicyTarget(LearningTarget):
    def __init__(self, name: str, loss_weight: float):
        super(PolicyTarget, self).__init__(name, loss_weight)

    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        The C++ uses the zero-tensor to represent a row that should be masked.
        """
        assert len(labels.shape) == 2, labels.shape
        n = labels.shape[0]
        labels = labels.reshape((n, -1))

        label_sums = labels.sum(dim=1)
        mask = label_sums != 0
        return mask

    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        selected_moves = torch.argmax(outputs, dim=1)
        correct_policy_preds = labels.gather(1, selected_moves.view(-1, 1))
        return int(sum(correct_policy_preds))


class ValueTarget(LearningTarget):
    def __init__(self, name: str, loss_weight: float):
        super(ValueTarget, self).__init__(name, loss_weight)

    def loss_fn(self) -> nn.Module:
        """
        AlphaGo, KataGo, etc., use the standard [-1, +1] range for value targets, and thus use MSE loss. In our project,
        however, in an effort to generalize to M-player games, we use a length-M vector of non-negative values
        summing to 1. So for us cross-entropy loss is more appropriate.

        NOTE: in conversations with David Wu, he suggested relaxing the fixed-utility assumption, to support games like
        Backgammon where the utility is not fixed. In that case, we would need to use MSE loss. There are some subtle
        c++ changes that would need to be made to support this.
        """
        return nn.CrossEntropyLoss()

    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        """
        Naively using the same implementation as PolicyTarget.get_num_correct_predictions() doesn't work for games that
        have draws. For example, in TicTacToe, if the true value is [0.5, 0.5], and the network outputs [0.4, 0.6], then
        the network should get credit for a correct prediction. But the naive implementation would give it no credit.

        Instead, in this example, we consider the output of [0.4, 0.6] to be correct, by virtue of the fact that
        abs([0.4, 0.6] - [0.5, 0.5]) = [0.1, 0.1] is "close enough" to [0, 0].
        """
        value_output_probs = outputs.softmax(dim=1)
        deltas = abs(value_output_probs - labels)
        return int(sum((deltas < 0.25).all(dim=1)))


class NeuralNet(nn.Module):
    _filename_to_net: Dict[str, torch.jit.ScriptModule] = {}

    def __init__(self, input_shape: Shape):
        """
        input_shape: the shape of a single row of data. The shape of the Tensor's that can passed into the
        forward() method will have one more dimension in front, corresponding to the number of rows.
        """
        super(NeuralNet, self).__init__()
        self.input_shape = input_shape

        # heads and learning_targets are parallel lists, with the same length
        self.heads = nn.ModuleList()
        self.learning_targets: List[LearningTarget] = []

    @classmethod
    def load_model(cls, filename: str, verbose: bool = False, eval_mode: bool = True) -> torch.jit.ScriptModule:
        """
        Loads a model previously saved to disk via save(). This uses torch.jit.load(), which returns a
        torch.jit.ScriptModule, which looks/feels/sounds like nn.Module, but is not exactly the same thing.

        For convenience, as a side-effect, this calls torch.set_grad_enabled(False), which mutates torch's global
        state.
        """
        net = NeuralNet._filename_to_net.get(filename, None)
        if net is None:
            if verbose:
                print(f'Loading model from {filename}')
            net = torch.jit.load(filename)
            if ENABLE_CUDA:
                net.to('cuda')
            if eval_mode:
                torch.set_grad_enabled(False)
                net.eval()
            else:
                net.train()
            NeuralNet._filename_to_net[filename] = net
            if verbose:
                print(f'Model successfully loaded!')
        return net

    def save_model(self, filename: str, verbose: bool = False):
        """
        Saves this network to disk, from which it can be loaded either by c++ or by python. Uses the
        torch.jit.trace() function to accomplish this.

        Note that prior to saving, we "freeze" the model, by switching it to eval mode and disabling gradient.
        The documentation seems to imply that this is an important step:

        "...In the returned :class:`ScriptModule`, operations that have different behaviors in ``training`` and
         ``eval`` modes will always behave as if it is in the mode it was in during tracing, no matter which mode the
          `ScriptModule` is in..."

        In order to avoid modifying self during the save() call, we actually deepcopy self and then do the freeze and
        trace on the copy.
        """
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        clone = copy.deepcopy(self)
        clone.to('cpu')
        clone.eval()
        forward_shape = tuple([1] + list(self.input_shape))
        example_input = torch.zeros(forward_shape)
        mod = torch.jit.trace(clone, example_input)
        mod.save(filename)
        if verbose:
            print(f'Model saved to {filename}')

    def add_head(self, head: nn.Module, target: LearningTarget):
        assert len(self.heads) == len(self.learning_targets)
        n = len(self.heads)
        if n == 0:
            assert target.name == 'policy', 'The first target must be the policy target'
        if n == 1:
            assert target.name == 'value', 'The second target must be the value target'

        assert target.name not in self.target_names(), f'Target with name {target.name} already exists'
        self.heads.append(head)
        self.learning_targets.append(target)

    def target_names(self) -> List[str]:
        return [target.name for target in self.learning_targets]

    @staticmethod
    @abc.abstractmethod
    def create(input_shape: Shape, head_names: List[str]) -> 'NeuralNet':
        """
        Create a new neural net of the derived type, with the given input shape, with the given heads.

        Different implementations of this method might differ in how they initialize architecture parameters (for
        example by loading from a config file). They also might differ in how they initialize model weights.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load_checkpoint(filename: str) -> 'NeuralNet':
        """
        Inverse of save_checkpoint(). Load a neural net from disk, so that it can be used for inference.
        """
        pass

    @abc.abstractmethod
    def save_checkpoint(self, filename: str):
        """
        Serialize the current state of this neural net to disk, so that it can be loaded later via load_checkpoint().
        """
        pass
