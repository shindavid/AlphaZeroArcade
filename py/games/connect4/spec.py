from dataclasses import dataclass
import math

from games.game_spec import GameSpec, ReferencePlayerFamily
from net_modules import ModelConfig, ModuleSpec
from util.torch_util import Shape


NUM_COLUMNS = 7
NUM_ROWS = 6
NUM_COLORS = 2
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


def b7_c64(input_shape: Shape):
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_shape = (NUM_COLUMNS, )
    c_trunk = 64
    c_mid = 64
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256
    c_ownership_hidden = 64

    return ModelConfig(
        input_shape=input_shape,

        stem=ModuleSpec(type='ConvBlock', args=[input_shape[0], c_trunk]),

        blocks=[
            ModuleSpec(type='ResBlock', args=['block1', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block2', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block3', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block4', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block5', c_trunk, c_mid]),

            ModuleSpec(type='ResBlock', args=['block6', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block7', c_trunk, c_mid]),
        ],

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden,
                             NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES]),
        ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'opp_policy': 0.15,
            'ownership': 0.15
        },
    )


@dataclass
class Connect4Spec(GameSpec):
    name = 'c4'
    extra_runtime_deps = ['extra_deps/connect4/c4solver',
                          'extra_deps/connect4/7x6.book']
    model_configs = {
        'default': b7_c64,
        'b7_c64': b7_c64,
    }
    reference_player_family = ReferencePlayerFamily('Perfect', '--strength', 0, 21)
    n_mcts_iters_for_ratings_matches = 100


Connect4 = Connect4Spec()
