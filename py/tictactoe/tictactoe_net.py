import math
from typing import List

import torch
from torch import nn as nn

from neural_net import NeuralNet, PolicyTarget, ValueTarget, OwnershipTarget
from res_net_modules import ConvBlock, ResBlock, PolicyHead, ValueHead, OwnershipHead
from util.torch_util import Shape


BOARD_LENGTH = 3
NUM_ACTIONS = BOARD_LENGTH * BOARD_LENGTH
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


class TicTacToeNet(NeuralNet):
    VALID_TARGET_NAMES = ['policy', 'value', 'opp_policy', 'ownership']

    def __init__(self, input_shape: Shape, target_names: List[str], n_conv_filters=64, n_res_blocks=9):
        for name in target_names:
            assert name in TicTacToeNet.VALID_TARGET_NAMES, name

        super(TicTacToeNet, self).__init__(input_shape)
        board_size = math.prod(input_shape[1:])
        self.n_conv_filters = n_conv_filters
        self.n_res_blocks = n_res_blocks
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])

        self.add_head(PolicyHead(board_size, NUM_ACTIONS, n_conv_filters), PolicyTarget('policy', 1.0))
        self.add_head(ValueHead(board_size, NUM_PLAYERS, n_conv_filters), ValueTarget('value', 1.5))
        if 'opp_policy' in target_names:
            self.add_head(PolicyHead(board_size, NUM_ACTIONS, n_conv_filters), PolicyTarget('opp_policy', 0.15))
        if 'ownership' in target_names:
            shape = (NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES, BOARD_LENGTH, BOARD_LENGTH)
            self.add_head(OwnershipHead(board_size, shape, n_conv_filters),
                          OwnershipTarget('ownership', 0.15))

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return tuple(head(x) for head in self.heads)

    @staticmethod
    def create(input_shape: Shape, target_names: List[str]) -> 'TicTacToeNet':
        """
        TODO: load architecture parameters from config and pass them to constructor call
        """
        return TicTacToeNet(input_shape, target_names)

    @staticmethod
    def load_checkpoint(filename: str) -> 'TicTacToeNet':
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']
        input_shape = checkpoint['input_shape']
        target_names = checkpoint['target_names']
        n_conv_filters = checkpoint['n_conv_filters']
        n_res_blocks = checkpoint['n_res_blocks']
        model = TicTacToeNet(input_shape, target_names, n_conv_filters, n_res_blocks)
        model.load_state_dict(model_state_dict)
        return model

    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'target_names': self.target_names(),
            'n_conv_filters': self.n_conv_filters,
            'n_res_blocks': self.n_res_blocks,
        }, filename)
