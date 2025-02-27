from alphazero.logic import constants

import argparse
from dataclasses import dataclass, fields
from typing import List


@dataclass
class LoopControllerParams:
    cuda_device: str = 'cuda:0'
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    model_cfg: str = 'default'
    target_rating_rate: float = 0.1
    ignore_sigint: bool = False

    @staticmethod
    def create(args) -> 'LoopControllerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(LoopControllerParams)}
        return LoopControllerParams(**kwargs)

    @staticmethod
    def add_args(parser, include_cuda_device=True):
        defaults = LoopControllerParams()
        group = parser.add_argument_group('LoopController options')

        if include_cuda_device:
            group.add_argument('--cuda-device', default=defaults.cuda_device,
                               help='cuda device used for network training (default: %(default)s)')
        group.add_argument('--port', type=int,
                           default=defaults.port,
                           help='LoopController port (default: %(default)s)')
        group.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                           help='model config (default: %(default)s)')
        group.add_argument('-R', '--target-rating-rate', type=float,
                           default=defaults.target_rating_rate,
                           help='target pct of generations that we want to rate. Ignored if at '
                           'least one rating server is using a dedicated GPU. Otherwise this '
                           'parameter is used to prevent rating servers from getting starved by '
                           'self-play/training. (default: %(default).1f)')
        group.add_argument('--ignore-sigint', action='store_true', default=defaults.ignore_sigint,
                           help=argparse.SUPPRESS)


    def add_to_cmd(self, cmd: List[str]):
        defaults = LoopControllerParams()
        for f in fields(LoopControllerParams):
            attr = getattr(self, f.name)
            if attr != getattr(defaults, f.name):
                cmd.append('--' + f.name)
                if type(attr) != bool:
                    cmd.append(str(getattr(self, f.name)))
