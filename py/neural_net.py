"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import copy
import os
from typing import Dict

import torch
from torch import nn as nn

from util.torch_util import Shape


ENABLE_CUDA = True


class NeuralNet(nn.Module):
    _filename_to_net: Dict[str, torch.jit.ScriptModule] = {}

    def __init__(self, input_shape: Shape):
        """
        input_shape: the shape of a single row of data. The shape of the Tensor's that can passed into the
        forward() method will have one more dimension in front, corresponding to the number of rows.
        """
        super(NeuralNet, self).__init__()
        self.input_shape = input_shape

    @classmethod
    def load(cls, filename: str, verbose: bool = False) -> torch.jit.ScriptModule:
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
            torch.set_grad_enabled(False)
            net.eval()
            NeuralNet._filename_to_net[filename] = net
            if verbose:
                print(f'Model successfully loaded!')
        return net

    def save(self, filename: str, verbose: bool = False):
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
