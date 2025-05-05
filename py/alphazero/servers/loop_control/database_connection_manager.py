from __future__ import annotations

from alphazero.logic import constants
from alphazero.logic.custom_types import ThreadId
from util.sqlite3_util import DatabaseConnectionPool

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


class DatabaseConnectionManager:
    def __init__(self, controller: LoopController):
        organizer = controller.organizer

        self.clients_db_conn_pool = DatabaseConnectionPool(
            organizer.clients_db_filename, constants.CLIENTS_TABLE_CREATE_CMDS)
        self.self_play_db_conn_pool = DatabaseConnectionPool(
            organizer.self_play_db_filename, constants.SELF_PLAY_TABLE_CREATE_CMDS)
        self.training_db_conn_pool = DatabaseConnectionPool(
            organizer.training_db_filename, constants.TRAINING_TABLE_CREATE_CMDS)
        self.ratings_db_conn_pool = DatabaseConnectionPool(
            organizer.ratings_db_filename, constants.RATINGS_TABLE_CREATE_CMDS)
        self.benchmark_db_conn_pool = DatabaseConnectionPool(
            organizer.benchmark_db_filename, constants.ARENA_TABLE_CREATE_CMDS)
        self.eval_db_conn_pool = DatabaseConnectionPool(
            organizer.eval_db_filename(controller.params.benchmark_tag), constants.ARENA_TABLE_CREATE_CMDS)

    def pools(self) -> List[DatabaseConnectionPool]:
        pools = [
            self.clients_db_conn_pool,
            self.self_play_db_conn_pool,
            self.training_db_conn_pool,
            self.ratings_db_conn_pool,
            self.benchmark_db_conn_pool,
            self.eval_db_conn_pool,
        ]
        return pools

    def close_db_conns(self, thread_id: ThreadId):
        for pool in self.pools():
            pool.close_connections(thread_id)
