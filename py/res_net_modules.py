import math
from typing import Union

from torch import nn as nn
from torch.nn import functional as F

from util.torch_util import Shape


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
    def __init__(self, board_size: int, policy_shape: Union[Shape, int], n_input_channels: int):
        super(PolicyHead, self).__init__()
        policy_shape = tuple([policy_shape]) if isinstance(policy_shape, int) else policy_shape
        self.board_size = board_size
        self.policy_shape = policy_shape
        self.policy_size = math.prod(policy_shape)

        self.conv = nn.Conv2d(n_input_channels, 2, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2 * board_size, self.policy_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, 2 * self.board_size)
        x = self.linear(x)
        x = x.view(-1, *self.policy_shape)
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

    Here, we are choosing to replace the scalar with a length-p array to generalize for p-player games. The output
    will be interpreted as logit probabilities for the corresponding player's expected win shares.
    """
    def __init__(self, board_size: int, n_players: int, n_input_channels: int):
        super(ValueHead, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(n_input_channels, 1, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(board_size, 256)
        self.linear2 = nn.Linear(256, n_players)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = x.view(-1, self.board_size)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ScoreMarginHead(nn.Module):
    def __init__(self, board_size: int, n_possible_score_margins: int, n_input_channels: int):
        super(ScoreMarginHead, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(n_input_channels, 1, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(board_size, 256)
        self.linear2 = nn.Linear(256, n_possible_score_margins)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = x.view(-1, self.board_size)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class OwnershipHead(nn.Module):
    def __init__(self, board_size: int, output_shape: Shape, n_input_channels: int):
        super(PolicyHead, self).__init__()
        self.board_size = board_size
        self.output_shape = output_shape
        self.output_size = math.prod(output_shape)

        self.conv = nn.Conv2d(n_input_channels, 2, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2 * board_size, self.output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, 2 * self.board_size)
        x = self.linear(x)
        x = x.view(-1, *self.output_shape)
        return x
