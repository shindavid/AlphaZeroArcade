import math
from typing import List

import numpy as np
import torch
from torch import nn as nn

from interface import NeuralNetworkInput, ActionIndex, AbstractGameTensorizor, AbstractSymmetryTransform, \
    IdentityTransform, PolicyTensor
from neural_net import NeuralNet
from profiling import ProfilerRegistry
from connect4.game_logic import C4GameState, NUM_COLUMNS, NUM_ROWS, NUM_COLORS, Color, MAX_MOVES_PER_GAME
from res_net_modules import ConvBlock, ResBlock, PolicyHead, ValueHead
from util.torch_util import Shape


# torch.set_default_dtype(torch.float64)


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
