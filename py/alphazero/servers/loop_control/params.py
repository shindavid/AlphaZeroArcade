from alphazero.logic import constants

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoopControllerParams:
    cuda_device: str = 'cuda:0'
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
    model_cfg: str = 'default'
    target_rating_rate: float = 0.1
    max_positions_per_generation: Optional[int] = None

    @staticmethod
    def create(args) -> 'LoopControllerParams':
        return LoopControllerParams(
            cuda_device=args.cuda_device,
            port=args.port,
            model_cfg=args.model_cfg,
            target_rating_rate=args.target_rating_rate,
            max_positions_per_generation=args.max_positions_per_generation,
        )

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
        group.add_argument('--max-positions-per-generation', type=int, default=None,
                           help='max number of positions per generation (default: None)')

    def add_to_cmd(self, cmd: List[str]):
        defaults = LoopControllerParams()
        if self.cuda_device != defaults.cuda_device:
            cmd.extend(['--cuda-device', self.cuda_device])
        if self.port != defaults.port:
            cmd.extend(['--port', str(self.port)])
        if self.model_cfg != defaults.model_cfg:
            cmd.extend(['--model-cfg', self.model_cfg])
        if self.target_rating_rate != defaults.target_rating_rate:
            cmd.extend(['--target-rating-rate', str(self.target_rating_rate)])
        if self.max_positions_per_generation != defaults.max_positions_per_generation:
            cmd.extend(['--max-positions-per-generation', str(self.max_positions_per_generation)])
