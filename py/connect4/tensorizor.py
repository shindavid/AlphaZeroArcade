from typing import List

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from interface import NeuralNetworkInput, ActionIndex, AbstractGameTensorizor, AbstractSymmetryTransform, \
    IdentityTransform, PolicyTensor
from neural_net import NeuralNet
from profiling import ProfilerRegistry
from connect4.game_logic import C4GameState, NUM_COLUMNS, NUM_ROWS, NUM_COLORS, Color, MAX_MOVES_PER_GAME
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

    • Input tensors X (shape b × b × cX) and G (shape b × b × cG).
    • A batch normalization layer and ReLu activation applied to G (output shape b × b × cG).
    • A global pooling layer (output shape 3cG).
    • A fully connected layer to cX outputs (output shape cX).
    • Channelwise addition with X (output shape b × b × cX).

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
        P: output of 1x1 convolution
        G: output of 1x1 convolution
        (Maybe X here becomes P?)
        GlobalPoolingBiasStruct pools G to bias the output of P
        """
        g = self.global_pool(self.relu(self.bn(g)))
        return self.fc(g[..., 0, 0])[..., None, None] + p



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

    Here, we are choosing to replace the scalar with a length-p array to generalize for p-player games. The output
    will be interpreted as logit probabilities for the corresponding player's expected win shares.
    """
    def __init__(self, n_input_channels: int):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, 1, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(NUM_COLUMNS * NUM_ROWS, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = x.view(-1, NUM_COLUMNS * NUM_ROWS)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class C4Net(NeuralNet):
    def __init__(self, input_shape: Shape, n_conv_filters=64, n_res_blocks=19):
        super(C4Net, self).__init__(input_shape)
        self.n_conv_filters = n_conv_filters
        self.n_res_blocks = n_res_blocks
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_conv_filters)
        self.value_head = ValueHead(n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'C4Net':
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


class HistoryBuffer:
    def __init__(self, num_previous_states: int):
        history_buffer_len = 1 + num_previous_states + MAX_MOVES_PER_GAME // NUM_COLORS
        shape = (NUM_COLORS, history_buffer_len, NUM_COLUMNS, NUM_ROWS)

        self.num_previous_states = num_previous_states
        self.full_mask = np.zeros(shape, dtype=bool)
        self.next_color = C4GameState.RED
        self.ref_indices = [self.num_previous_states, self.num_previous_states]

    def update(self, game: C4GameState):
        ProfilerRegistry['histbuff.upd'].start()
        self.ref_indices[self.next_color] += 1
        self.full_mask[self.next_color][self.ref_indices[self.next_color]] = game.get_mask(self.next_color)
        self.next_color = self.prev_color
        assert self.next_color == game.get_current_player()
        ProfilerRegistry['histbuff.upd'].stop()

    def undo(self):
        """
        Undo last update() call.
        """
        self.next_color = self.prev_color
        self.ref_indices[self.next_color] -= 1

    @property
    def prev_color(self) -> Color:
        return 1 - self.next_color

    def get_input(self):
        n = self.next_color
        p = self.prev_color
        ni = self.ref_indices[n]
        pi = self.ref_indices[p]
        n_mask = self.full_mask[n][ni-self.num_previous_states:ni+1]
        p_mask = self.full_mask[p][pi-self.num_previous_states:pi+1]
        return np.concatenate((n_mask, p_mask))

    def get_shape(self) -> Shape:
        return self.num_previous_states*2+2, NUM_COLUMNS, NUM_ROWS


class C4ReflectionTransform(AbstractSymmetryTransform):
    def transform_input(self, net_input: NeuralNetworkInput) -> NeuralNetworkInput:
        return torch.flip(net_input, (2, ))

    def transform_policy(self, policy: PolicyTensor) -> PolicyTensor:
        return torch.flip(policy, (0,))


class C4Tensorizor(AbstractGameTensorizor):
    def __init__(self, num_previous_states: int):
        self.num_previous_states = num_previous_states
        self.history_buffer = HistoryBuffer(num_previous_states)
        self.move_stack: List[ActionIndex] = []
        self.symmetries = [IdentityTransform(), C4ReflectionTransform()]

    def clear(self):
        self.history_buffer = HistoryBuffer(self.num_previous_states)
        self.move_stack = []

    @staticmethod
    def get_num_previous_states(shape: Shape) -> int:
        return (shape[0] - 2) // 2

    @staticmethod
    def get_input_shape(num_previous_states) -> Shape:
        return num_previous_states*2+2, NUM_COLUMNS, NUM_ROWS

    def receive_state_change(self, state: C4GameState, action_index: ActionIndex):
        self.history_buffer.update(state)
        self.move_stack.append(action_index)

    @staticmethod
    def supports_undo() -> bool:
        return True

    def undo(self, state: C4GameState):
        move = self.move_stack.pop()
        state.undo_move(move+1)
        self.history_buffer.undo()

    def vectorize(self, state: C4GameState) -> NeuralNetworkInput:
        ProfilerRegistry['vectorize'].start()
        i = self.history_buffer.get_input()
        shape = i.shape
        tensor_shape = tuple([1] + list(shape))
        v = torch.reshape(torch.from_numpy(i), tensor_shape).float()
        ProfilerRegistry['vectorize'].stop()
        return v

    def get_symmetries(self, state: C4GameState) -> List[AbstractSymmetryTransform]:
        return self.symmetries
