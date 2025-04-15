from alphazero.logic.agent_types import MCTSAgent
from alphazero.logic.build_params import BuildParams
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.base_server import BaseServer, ServerConstants
from util.logging_util import LoggingParams
from util.socket_util import JsonDict
from util import subprocess_util
from util.str_util import make_args_str

from dataclasses import dataclass, fields
import logging
import os
from typing import Optional, Dict


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkServerParams(BaseParams):
    rating_tag: str = ''

    @staticmethod
    def create(args) -> 'BenchmarkServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(BenchmarkServerParams)}
        return BenchmarkServerParams(**kwargs)

    @staticmethod
    def add_args(parser, omit_base=False):
        defaults = BenchmarkServerParams()

        group = parser.add_argument_group(f'EvalServer options')
        if not omit_base:
            BaseParams.add_base_args(group)

        group.add_argument('-r', '--rating-tag', default=defaults.rating_tag,
                           help='evaluation tag. Loop controller collates ratings by this str. It is '
                           'the responsibility of the user to make sure that the same '
                           'binary/params are used across different EvalServer processes '
                           'sharing the same rating-tag. (default: "%(default)s")')


class BenchmarkServer(BaseServer):
    SERVER_CONSTANTS = ServerConstants(
        server_name='benchmark-server',
        worker_name='benchmark-worker',
        server_role=ClientRole.BENCHMARK_SERVER,
        worker_role=ClientRole.BENCHMARK_WORKER)

    def __init__(self, params: BenchmarkServerParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        super().__init__(params, logging_params, build_params)

    def _send_handshake(self):
        self._session_data.send_handshake(ClientRole.BENCHMARK_SERVER)
