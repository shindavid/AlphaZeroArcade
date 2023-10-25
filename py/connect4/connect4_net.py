import math

from res_net_modules import ModelConfig, ModuleSpec, GlobalPoolingLayer
from util.torch_util import Shape


NUM_COLUMNS = 7
NUM_ROWS = 6
NUM_COLORS = 2
NUM_PLAYERS = 2
NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES = NUM_PLAYERS + 1  # +1 for empty square


def c4_b19_c64(input_shape: Shape):
    board_shape = input_shape[1:]
    board_size = math.prod(board_shape)
    policy_shape = (NUM_COLUMNS, )
    c_trunk = 64
    c_mid = 64
    c_neck_spatial = 64
    c_neck_gpool = 64
    c_neck_gpool_out = GlobalPoolingLayer.NUM_CHANNELS * c_neck_gpool

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

        neck=ModuleSpec(type='Neck', args=[c_trunk, c_neck_spatial, c_neck_gpool]),

        heads=[
            ModuleSpec(type='PolicyHead',
                       args=['policy', board_size, c_neck_spatial, policy_shape]),
            ModuleSpec(type='ValueHead', args=['value', c_neck_gpool_out, NUM_COLORS]),
            ModuleSpec(type='PolicyHead',
                       args=['opp_policy', board_size, c_neck_spatial, policy_shape]),
            ModuleSpec(type='OwnershipHead',
                       args=['ownership', c_neck_spatial, NUM_POSSIBLE_END_OF_GAME_SQUARE_STATES]),
            ],

        loss_weights={
            'policy': 1.0,
            'value': 1.5,
            'opp_policy': 0.15,
            'ownership': 0.15
            },
        )


C4_MODEL_CONFIGS = {
    'c4_b19_c64': c4_b19_c64,
    'default': c4_b19_c64,
}
