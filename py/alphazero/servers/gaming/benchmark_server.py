from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole
from alphazero.servers.gaming.server_base import ServerBase, ServerConfig, ServerParams
from shared.rating_params import RatingParams
from util.logging_util import LoggingParams


class BenchmarkServer(ServerBase):
    def __init__(self, params: ServerParams, logging_params: LoggingParams,
                 build_params: BuildParams, rating_params: RatingParams):
        server_config = ServerConfig(
            server_name='benchmark-server',
            worker_name='benchmark-worker',
            server_role=ClientRole.BENCHMARK_SERVER,
            worker_role=ClientRole.BENCHMARK_WORKER)
        super().__init__(params, logging_params, build_params, rating_params, server_config)
