from alphazero.logic import constants

from dataclasses import dataclass
from typing import List


@dataclass
class LoopControllerParams:
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    cuda_device: str = 'cuda:0'
    model_cfg: str = 'default'

    @staticmethod
    def create(args) -> 'LoopControllerParams':
        return LoopControllerParams(
            port=args.port,
            cuda_device=args.cuda_device,
            model_cfg=args.model_cfg,
        )

    @staticmethod
    def add_args(parser):
        defaults = LoopControllerParams()
        group = parser.add_argument_group('LoopController options')

        group.add_argument('--port', type=int,
                           default=defaults.port,
                           help='LoopController port (default: %(default)s)')
        group.add_argument('--cuda-device',
                           default=defaults.cuda_device,
                           help='cuda device used for network training (default: %(default)s)')
        group.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                           help='model config (default: %(default)s)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = LoopControllerParams()
        if self.port != defaults.port:
            cmd.extend(['--port', str(self.port)])
        if self.cuda_device != defaults.cuda_device:
            cmd.extend(['--cuda-device', self.cuda_device])
        if self.model_cfg != defaults.model_cfg:
            cmd.extend(['--model-cfg', self.model_cfg])