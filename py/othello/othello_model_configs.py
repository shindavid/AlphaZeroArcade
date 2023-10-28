import math

from net_modules import ModelConfig, ModuleSpec
from util.torch_util import Shape


BOARD_LENGTH = 8
NUM_SQUARES = BOARD_LENGTH * BOARD_LENGTH
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


def othello_b19_c64(input_shape: Shape):
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_shape = (NUM_SQUARES,)
    c_trunk = 64
    c_mid = 64
    c_policy_hidden = 2
    c_opp_policy_hidden = 2
    c_value_hidden = 1
    n_value_hidden = 256
    c_score_margin_hidden = 1
    n_score_margin_hidden = 256
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
            ModuleSpec(type='ResBlock', args=['block8', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block9', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block10', c_trunk, c_mid]),

            ModuleSpec(type='ResBlock', args=['block11', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block12', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block13', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block14', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block15', c_trunk, c_mid]),

            ModuleSpec(type='ResBlock', args=['block16', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block17', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block18', c_trunk, c_mid]),
            ModuleSpec(type='ResBlock', args=['block19', c_trunk, c_mid]),
            ],

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_trunk, c_policy_hidden, policy_shape]),
            ModuleSpec(type='ValueHead',
                       args=['value', board_size, c_trunk, c_value_hidden, n_value_hidden,
                             NUM_PLAYERS]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_trunk, c_opp_policy_hidden, policy_shape]),
            ModuleSpec(type='ScoreMarginHead',
                       args=['score_margin', board_size, c_trunk, c_score_margin_hidden,
                             n_score_margin_hidden, NUM_SQUARES]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_trunk, c_ownership_hidden,
                             NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES]),
            ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'opp_policy': 0.15,
            'score_margin': 0.02,
            'ownership': 0.15
            },
        )


OTHELLO_MODEL_CONFIGS = {
    'othello_b19_c64': othello_b19_c64,
    'default': othello_b19_c64,
}
