from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole
from alphazero.servers.gaming.server_base import ServerBase, ServerConfig, ServerParams
from util.logging_util import LoggingParams

class EvalServerParams(ServerParams):
    SERVER_NAME = 'eval-server'

class EvalServer(ServerBase):
    def __init__(self, params: ServerParams, logging_params: LoggingParams, build_params: BuildParams):
        server_config = ServerConfig(
        server_name='eval-server',
        worker_name='eval-worker',
        server_role=ClientRole.EVAL_SERVER,
        worker_role=ClientRole.EVAL_WORKER)
        super().__init__(params, logging_params, build_params, server_config)