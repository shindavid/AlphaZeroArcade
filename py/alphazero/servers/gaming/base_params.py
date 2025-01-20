from alphazero.logic import constants

import argparse
from dataclasses import dataclass


@dataclass
class BaseParams:
    loop_controller_host: str = 'localhost'
    loop_controller_port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'
    ignore_sigint: bool = False

    @staticmethod
    def add_base_args(group):
        defaults = BaseParams()

        group.add_argument('--loop-controller-host', type=str,
                           default=defaults.loop_controller_host,
                           help='loop-controller host (default: %(default)s)')
        group.add_argument('--loop-controller-port', type=int,
                           default=defaults.loop_controller_port,
                           help='loop-controller port (default: %(default)s)')
        group.add_argument('--cuda-device', default=defaults.cuda_device,
                           help='cuda device (default: %(default)s)')
        group.add_argument('--ignore-sigint', action='store_true', help=argparse.SUPPRESS)

        return group
