from .loop_controller_interface import LoopControllerInterface

from alphazero.logic import constants
from alphazero.logic.custom_types import ThreadId
from util.sqlite3_util import DatabaseConnectionPool


class DatabaseConnectionManager:
    def __init__(self, controller: LoopControllerInterface):
        organizer = controller.organizer

        self.clients_db_conn_pool = DatabaseConnectionPool(
            organizer.clients_db_filename, constants.CLIENTS_TABLE_CREATE_CMDS)
        self.self_play_db_conn_pool = DatabaseConnectionPool(
            organizer.self_play_db_filename, constants.SELF_PLAY_TABLE_CREATE_CMDS)
        self.training_db_conn_pool = DatabaseConnectionPool(
            organizer.training_db_filename, constants.TRAINING_TABLE_CREATE_CMDS)
        self.ratings_db_conn_pool = DatabaseConnectionPool(
            organizer.ratings_db_filename, constants.RATINGS_TABLE_CREATE_CMDS)

    def _pools(self):
        return [self.clients_db_conn_pool, self.self_play_db_conn_pool, self.training_db_conn_pool]

    def close_db_conns(self, thread_id: ThreadId):
        for pool in self._pools():
            pool.close_connections(thread_id)
