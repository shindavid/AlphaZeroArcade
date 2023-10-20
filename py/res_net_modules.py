import math
from typing import Union
import torch
from torch import nn as nn
from torch.nn import functional as F

from util.torch_util import Shape


class GlobalPoolingLayer(nn.Module):
    """
    From "Accelerating Self-Play Learning in Go" (KataGo paper):
    The Global Pooling module as described in the paper:
    1. The mean of each channel
    2. The mean of each channel multiplied by 1/10 ( b - b_avg )
    3. The maximum of each channel.
    https://arxiv.org/pdf/1902.10565.pdf
    """
    def __init__(self, scale=1/10):
        super(GlobalPoolingLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        g_mean = torch.mean(x, keepdim=True, dim=(2, 3))
        g_scaled_mean = self.scale * g_mean
        # g_max shape: NC1
        g_max, _ = torch.max(x.view(x.shape[:2] + (-1,)), dim=-1, keepdim=True)
        return torch.cat([g_mean, g_scaled_mean, g_max[..., None]], dim=1)


class GlobalPoolingBiasStruct(nn.Module):
    """
    From "Accelerating Self-Play Learning in Go" (KataGo paper):
    The  global pooling bias structure as described in the paper:
    • Input tensors X (shape b x b x cX) and G (shape b x b x cG).
    • A batch normalization layer and ReLu activation applied to G (output shape b x b x cG).
    • A global pooling layer (output shape 3cG).
    • A fully connected layer to cX outputs (output shape cX).
    • Channelwise addition with X (output shape b x b x cX).
    https://arxiv.org/pdf/1902.10565.pdf
    """
    def __init__(self, g_channels):
        super(GlobalPoolingBiasStruct, self).__init__()
        self.global_pool = GlobalPoolingLayer()
        self.bn = nn.BatchNorm2d(g_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3 * g_channels, g_channels)

    def forward(self, p, g):
        """
        (My understanding*) For the policy head:
        X: the board feature map (b x b x cx)
        P: output of 1x1 convolution ?
        G: output of 1x1 convolution ?
        (Maybe X here becomes P?)
        GlobalPoolingBiasStruct pools G to bias the output of P
        """
        g = self.global_pool(self.relu(self.bn(g)))
        g = self.fc(g[..., 0, 0])[..., None, None]
        return p + g


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


class GPResBlock(nn.Module):
    """
    From "Accelerating Self-Play Learning in Go" (KataGo paper):
    The residual block with global pooling bias structure as described in the paper
    https://arxiv.org/pdf/1902.10565.pdf
    """
    def __init__(self, n_conv_filters: int):
        super(GPResBlock, self).__init__()
        assert n_conv_filters % 2 == 0, 'n_conv_filters has to be even'
        self.c_mid = n_conv_filters // 2
        self.gpbs = GlobalPoolingBiasStruct(self.c_mid)
        self.conv1 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(n_conv_filters)
        self.conv2 = nn.Conv2d(self.c_mid, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        identity = x
        # outputs [N, C, H, W]
        out = F.relu(self.batch1(self.conv1(x)))
        # use first c_pool layers, to bias the the other part 
        # outputs [N, Cp, H, W]
        # return torch.zeros(0)
        out = self.gpbs(out[:, self.c_mid:], out[:, :self.c_mid])
        # outputs [N, C, H, W]
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

    According to the Oracle blog post, both the Oracle dev team and the LeelaChess dev team found
    that increasing the number of filters to 32 in the output head significantly sped up
    training, so we do the same here.
    """
    def __init__(self, board_size: int, policy_shape: Union[Shape, int], n_input_channels: int,
                 n_filters=2):
        super(PolicyHead, self).__init__()
        policy_shape = tuple([policy_shape]) if isinstance(policy_shape, int) else policy_shape
        self.board_size = board_size
        self.policy_shape = policy_shape
        self.policy_size = math.prod(policy_shape)
        self.n_filters = n_filters

        self.conv = nn.Conv2d(n_input_channels, n_filters, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(n_filters)
        self.linear = nn.Linear(n_filters * board_size, self.policy_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, self.n_filters * self.board_size)
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

    According to the Oracle blog post, both the Oracle dev team and the LeelaChess dev team found
    that increasing the number of filters to 32 in the output head significantly sped up
    training, so we do the same here.
    """
    def __init__(self, board_size: int, n_players: int, n_input_channels: int,
                 n_filters=1):
        super(ValueHead, self).__init__()
        self.board_size = board_size
        self.n_filters = n_filters

        self.conv = nn.Conv2d(n_input_channels, n_filters, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(n_filters)
        self.linear1 = nn.Linear(n_filters * board_size, 256)
        self.linear2 = nn.Linear(256, n_players)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = x.view(-1, self.n_filters * self.board_size)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ScoreMarginHead(nn.Module):
    def __init__(self, board_size: int, n_possible_score_margins: int, n_input_channels: int,
                 n_filters=1):
        super(ScoreMarginHead, self).__init__()
        self.board_size = board_size
        self.n_filters = n_filters
        self.conv = nn.Conv2d(n_input_channels, n_filters, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(n_filters)
        self.linear1 = nn.Linear(board_size * n_filters, 256)
        self.linear2 = nn.Linear(256, n_possible_score_margins)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = x.view(-1, self.n_filters * self.board_size)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class OwnershipHead(nn.Module):
    def __init__(self, board_size: int, output_shape: Shape, n_input_channels: int,
                 n_filters=2):
        super(OwnershipHead, self).__init__()
        self.board_size = board_size
        self.n_filters = n_filters
        self.output_shape = output_shape
        self.output_size = math.prod(output_shape)

        self.conv = nn.Conv2d(n_input_channels, n_filters, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(n_filters)
        self.linear = nn.Linear(n_filters * board_size, self.output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, self.n_filters * self.board_size)
        x = self.linear(x)
        x = x.view(-1, *self.output_shape)
        return x
