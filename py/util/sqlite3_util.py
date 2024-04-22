from util.logging_util import get_logger

import os
import sqlite3
import threading
from typing import Dict, List, Optional


logger = get_logger()


class DatabaseConnectionPool:
    """
    This class is used to manage connections to a sqlite3 database in a multi-threaded environment.
    It should be constructed in the main thread. Subsequently spawned threads can call
    get_connection() to get a connection to the database. Repeated calls to get_connection() from
    the same thread will return the same connection object.

    This class is necessary because sqlite3 only allows one database connection per thread.

    This class also provides a lock (db_lock) that can be used to synchronize access to the
    database. Usage of this lock is optional.
    """
    def __init__(self, db_filename: str, create_cmds: Optional[List[str]]=None):
        """
        create_cmds is a list of SQL commands to create the tables in the database. It is used to
        create the table if the file does not exist. If not specified, and if db_filename does not
        exist, then an exception is raised.
        """
        self._db_filename = db_filename
        self._db_exists = None
        self._create_cmds = create_cmds
        self._conn_dict: Dict[int, sqlite3.Connection] = {}  # thread-id -> connection
        self._conn_dict_lock = threading.Lock()
        self._db_lock = threading.Lock()

    @property
    def db_lock(self):
        return self._db_lock

    def get_cursor(self) -> sqlite3.Cursor:
        return self.get_connection().cursor()

    def close_connections(self, thread_id=None):
        """
        Close all connections for the given thread. If thread_id is not specified, then the
        current thread's id is used.
        """
        if thread_id is None:
            thread_id = threading.get_ident()
        with self._conn_dict_lock:
            conn = self._conn_dict.pop(thread_id, None)
            if conn:
                conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """
        Returns a connection to the database. Each thread gets its own connection.
        """
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        with self._conn_dict_lock:
            conn = self._conn_dict.get(thread_id, None)
            if conn is None:
                n = len(self._conn_dict) + 1
                logger.debug(f"Creating new connection: db_filename={self._db_filename} thread={thread_name} ({n})")
                conn = self._create_connection()
                self._conn_dict[thread_id] = conn
            return conn

    def _create_connection(self) -> sqlite3.Connection:
        """
        Assumes that self._conn_dict_lock is already acquired.
        """
        if not self._db_exists:
            if not os.path.isfile(self._db_filename):
                if not self._create_cmds:
                    raise ValueError(f"Database file {self._db_filename} does not exist and create_cmds is not specified.")

                create_conn = sqlite3.connect(self._db_filename)
                cursor = create_conn.cursor()
                for cmd in self._create_cmds:
                    cursor.execute(cmd)
                create_conn.commit()
                create_conn.close()
            self._db_exists = True

        conn = sqlite3.connect(self._db_filename)
        return conn


def copy_db(source_filename: str, target_filename: str, where_clause: str):
    logger.info(f"Copying database from {source_filename} to {target_filename} "
                f"with where clause: {where_clause}")
    conn = sqlite3.connect(source_filename)
    conn.execute(f"ATTACH DATABASE '{target_filename}' AS target_db")

    cursor = conn.cursor()

    # First, copy tables
    cursor.execute(f'SELECT name, sql FROM sqlite_master WHERE type="table"')
    for table, sql in cursor.fetchall():
        if table == 'sqlite_sequence':
            continue

        logger.info(f"Copying table {table}...")
        sql_target = sql.replace('CREATE TABLE', 'CREATE TABLE target_db.')
        conn.execute(sql_target)
        conn.execute(
            f"INSERT INTO target_db.{table} SELECT * FROM main.{table} WHERE {where_clause}")

    # Next, copy triggers
    cursor.execute(f'SELECT name, sql FROM sqlite_master WHERE type="trigger"')
    for name, sql in cursor.fetchall():
        logger.info(f"Copying trigger {name}...")
        sql_target = sql.replace('CREATE TRIGGER ', f'CREATE TRIGGER target_db.')
        conn.execute(sql_target)

    conn.commit()
    conn.close()

    logger.info('Database copy complete.')
