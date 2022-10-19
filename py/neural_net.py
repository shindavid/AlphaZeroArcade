"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import os
# from functools import lru_cache
from typing import Dict

import torch
from torch import nn as nn, Tensor

from config import Config


ENABLE_CUDA = True


class NeuralNet(nn.Module):
    LRU_CACHE_SIZE = Config.instance().get('net.lru_cache.size', 65536)
    _filename_to_net: Dict[str, 'NeuralNet'] = {}

    def __init__(self, constructor_args):
        super(NeuralNet, self).__init__()
        self.constructor_args = constructor_args

    # @lru_cache(LRU_CACHE_SIZE)  # doesn't work, hash(Tensor) is based on id()
    def __call__(self, tensor: Tensor):
        out = super(NeuralNet, self).__call__(tensor)
        return out

    @classmethod
    def load(cls, filename: str, verbose: bool = False) -> 'NeuralNet':
        net = NeuralNet._filename_to_net.get(filename, None)
        if net is None:
            if verbose:
                print(f'Loading model from {filename}')
            model_data = torch.load(filename)
            net = cls(*model_data['model.constructor_args'])
            net.load_state_dict(model_data['model.state_dict'])
            if ENABLE_CUDA:
                net.to('cuda')
            torch.set_grad_enabled(False)
            net.eval()
            NeuralNet._filename_to_net[filename] = net
            if verbose:
                print(f'Model successfully loaded!')
        return net

    def save(self, filename: str, verbose: bool = False):
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model.constructor_args': self.constructor_args,
            'model.state_dict': self.state_dict(),
        }, filename)
        if verbose:
            print(f'Model saved to {filename}')
