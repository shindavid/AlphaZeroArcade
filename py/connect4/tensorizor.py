import math

import torch
from torch import nn as nn

from neural_net import NeuralNet
from res_net_modules import ConvBlock, ResBlock, PolicyHead, ValueHead
from util.torch_util import Shape


NUM_COLUMNS = 7
NUM_COLORS = 2


class C4Net(NeuralNet):
    def __init__(self, input_shape: Shape, n_conv_filters=64, n_res_blocks=19):
        super(C4Net, self).__init__(input_shape)
        board_size = math.prod(input_shape[1:])
        self.n_conv_filters = n_conv_filters
        self.n_res_blocks = n_res_blocks
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(board_size, NUM_COLUMNS, n_conv_filters)
        self.value_head = ValueHead(board_size, NUM_COLORS, n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

    @staticmethod
    def create(input_shape: Shape) -> 'C4Net':
        """
        TODO: load architecture parameters from config and pass them to constructor call
        """
        return C4Net(input_shape)

    @staticmethod
    def load_checkpoint(filename: str) -> 'C4Net':
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']
        input_shape = checkpoint['input_shape']
        n_conv_filters = checkpoint['n_conv_filters']
        n_res_blocks = checkpoint['n_res_blocks']
        model = C4Net(input_shape, n_conv_filters, n_res_blocks)
        model.load_state_dict(model_state_dict)
        return model

    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'n_conv_filters': self.n_conv_filters,
            'n_res_blocks': self.n_res_blocks,
        }, filename)
