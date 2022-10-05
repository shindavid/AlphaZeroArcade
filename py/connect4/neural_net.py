import os
import sys
from typing import Tuple, Optional, Hashable
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from game_logic import Game, NUM_COLUMNS, NUM_ROWS
from game_logic import NUM_COLORS, Color, MAX_MOVES_PER_GAME

sys.path.append(os.path.join(sys.path[0], '..'))
from interface import AbstractNeuralNetwork, NeuralNetworkInput, GlobalPolicyLogitDistr, ValueLogitDistr, \
    AbstractGameState, PlayerIndex, ActionIndex, ActionMask, ValueProbDistr


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
    def __init__(self, n_input_channels: int):
        super(ValueHead, self).__init__()
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


class Net(nn.Module):
    def __init__(self, input_shape: Shape, n_conv_filters=64, n_res_blocks=19):
        super(Net, self).__init__()
        self.constructor_args = (input_shape, n_conv_filters, n_res_blocks)  # to aid in loading of saved model object
        self.input_shape = input_shape
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_conv_filters)
        self.value_head = ValueHead(n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)


class HistoryBuffer:
    def __init__(self, num_previous_states: int):
        history_buffer_len = 1 + num_previous_states + MAX_MOVES_PER_GAME // NUM_COLORS
        shape = (NUM_COLORS, history_buffer_len, NUM_COLUMNS, NUM_ROWS)

        self.num_previous_states = num_previous_states
        self.full_mask = np.zeros(shape, dtype=bool)
        self.next_color = Game.RED
        self.ref_indices = [self.num_previous_states, self.num_previous_states]

    def update(self, game: Game):
        self.ref_indices[self.next_color] += 1
        self.full_mask[self.next_color][self.ref_indices[self.next_color]] = game.get_mask(self.next_color)
        self.next_color = self.prev_color
        assert self.next_color == game.get_current_player()

    def undo(self):
        """
        Undo last update() call.
        """
        self.next_color = self.prev_color
        self.full_mask[self.next_color][self.ref_indices[self.next_color]] = 0
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

    @staticmethod
    def get_shape(num_previous_states: int) -> Shape:
        return num_previous_states*2+2, NUM_COLUMNS, NUM_ROWS

    @staticmethod
    def get_num_previous_states(shape: Shape) -> int:
        return (shape[0] - 2) // 2


class NetWrapper(AbstractNeuralNetwork):
    def __init__(self, net: Net):
        self.net = net

    def evaluate(self, vec: NeuralNetworkInput) -> Tuple[GlobalPolicyLogitDistr, ValueLogitDistr]:
        p, v = self.net(vec)
        return p.flatten(), v.flatten()


class GameState(AbstractGameState):
    def __init__(self, game: Game, history_buffer: HistoryBuffer):
        self.winners = []
        self.game = game
        self.history_buffer = history_buffer
        self.move_stack = []

    @staticmethod
    def supports_undo() -> bool:
        return True

    @staticmethod
    def get_num_global_actions() -> int:
        return NUM_COLUMNS

    def debug_dump(self, file_handle):
        file_handle.write(self.game.to_ascii_drawing(pretty_print=False))

    def compact_repr(self) -> str:
        return self.game.get_board_str()

    def to_xml_tree(self, elem: ET.Element, tag: str) -> ET.Element:
        tag_dict = {
            'board': self.game.get_board_str(),
        }
        return ET.SubElement(elem, tag, tag_dict)

    def get_signature(self) -> Hashable:
        return self.game.piece_mask.data.tobytes(), self.game.current_player

    def get_current_player(self) -> PlayerIndex:
        return self.game.get_current_player()

    def apply_move(self, action_index: ActionIndex):
        self.winners = self.game.apply_move(action_index+1)
        self.history_buffer.update(self.game)
        self.move_stack.append(action_index)

    def undo_last_move(self):
        action_index = self.move_stack.pop()
        self.history_buffer.undo()
        self.winners = []
        self.game.undo_move(action_index+1)

    def get_valid_actions(self) -> ActionMask:
        actions = np.array(self.game.get_valid_moves())
        mask = torch.zeros(self.get_num_global_actions(), dtype=bool)
        mask[actions-1] = 1
        return mask

    def get_game_result(self) -> Optional[ValueProbDistr]:
        if self.winners:
            arr = np.zeros(2)
            arr[self.winners] = 1.0 / len(self.winners)
            return arr
        return None

    def vectorize(self) -> NeuralNetworkInput:
        i = self.history_buffer.get_input()
        shape = i.shape
        tensor_shape = tuple([1] + list(shape))
        return torch.reshape(torch.from_numpy(i), tensor_shape).float()
