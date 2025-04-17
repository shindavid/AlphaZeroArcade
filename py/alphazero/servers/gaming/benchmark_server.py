from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole
from alphazero.servers.gaming.server_base import ServerBase, ServerConfig, ServerParams
from util.logging_util import LoggingParams

class BenchmarkServerParams(ServerParams):
    SERVER_NAME = 'benchmark-server'

class BenchmarkServer(ServerBase):
    def __init__(self, params: ServerParams, logging_params: LoggingParams, build_params: BuildParams):
        server_config = ServerConfig(
            server_name='benchmark-server',
            worker_name='benchmark-worker',
            server_role=ClientRole.BENCHMARK_SERVER,
            worker_role=ClientRole.BENCHMARK_WORKER)
        super().__init__(params, logging_params, build_params, server_config)