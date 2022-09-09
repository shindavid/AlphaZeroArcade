from typing import Tuple

import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from game import Game, NUM_COLUMNS, NUM_ROWS

Shape = Tuple[int, ...]


class ConvBlock(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The convolutional block applies the following modules:

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_input_channels: int, n_conv_filters: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        return F.relu(self.batch(self.conv(x)))


class ResBlock(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    Each residual block applies the following modules sequentially to its input:

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_conv_filters: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(n_conv_filters)
        self.conv2 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        identity = x
        out = F.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        out += identity  # skip connection
        return F.relu(out)


class PolicyHead(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The policy head applies the following modules:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19^2 + 1 = 362 corresponding to
    logit probabilities for all intersections and the pass move

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_input_channels: int):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, 2, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2 * NUM_COLUMNS * NUM_ROWS, NUM_COLUMNS)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, 2 * NUM_COLUMNS * NUM_ROWS)
        x = self.linear(x)
        return x


class ValueHead(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The value head applies the following modules:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [−1, 1]

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self):
        super(ValueHead, self).__init__()
        raise Exception('TODO')

    def forward(self, x):
        pass


class Net(nn.Module):
    def __init__(self, input_shape: Shape, n_conv_filters=64, n_res_blocks=19):
        super(Net, self).__init__()
        self.constructor_args = (input_shape, n_conv_filters, n_res_blocks)  # to aid in loading of saved model object
        self.input_shape = input_shape
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x)


class InputBuilder:
    def __init__(self, num_previous_states: int):
        self.num_previous_states = num_previous_states

        self.full_red_mask = np.zeros((num_previous_states + 1, NUM_COLUMNS, NUM_ROWS), dtype=bool)
        self.full_yellow_mask = np.zeros((num_previous_states + 1, NUM_COLUMNS, NUM_ROWS), dtype=bool)

    def get_shape(self):
        return (self.num_previous_states*2+3, NUM_COLUMNS, NUM_ROWS)

    def start_game(self):
        """
        Should be called once at start of game.
        """
        self.full_red_mask &= False
        self.full_yellow_mask &= False

    def get_input(self, g: Game) -> np.ndarray:
        """
        Should be called after each move made in the game.

        Returns a 3-D numpy array.
        """
        shape1 = (1, NUM_COLUMNS, NUM_ROWS)
        cur_player = g.get_current_player()
        cur_player_mask = np.zeros(shape1, dtype=bool) + cur_player
        if cur_player == Game.RED:
            yellow_mask = g.get_mask(Game.YELLOW)
            self.full_yellow_mask = np.concatenate((yellow_mask.reshape(shape1), self.full_yellow_mask[:-1]))
        else:
            red_mask = g.get_mask(Game.RED)
            self.full_red_mask = np.concatenate((red_mask.reshape(shape1), self.full_red_mask[:-1]))

        return np.concatenate((self.full_red_mask, self.full_yellow_mask, cur_player_mask))
