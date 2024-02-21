from util.logging_util import get_logger

import os
import sqlite3
import threading
from typing import Dict, List, Optional


logger = get_logger()


class MultiThreadedConnectionManager:
    """
    This class is used to manage a sqlite3 database connection in a multi-threaded environment. It
    should be constructed in the main thread. Subsequently spawned threads can call get_connection()
    to get a connection to the database. Repeated calls to get_connection() from the same thread
    will return the same connection object.

    This class is necessary because sqlite3 only allows one database connection per thread.
    """
    def __init__(self, db_filename: str, create_cmds: Optional[List[str]]=None):
        """
        create_cmds is a list of SQL commands to create the tables in the database. It is used to
        create the table if the file does not exist. If not specified, and if db_filename does not
        exist, then an exception is raised.
        """
        self._db_filename = db_filename
        self._create_cmds = create_cmds
        self._conn_dict_readonly: Dict[int, sqlite3.Connection] = {}  # thread-id -> connection
        self._conn_dict_readwrite: Dict[int, sqlite3.Connection] = {}  # thread-id -> connection
        self._conn_dict_lock = threading.Lock()

    def get_cursor(self, readonly=True) -> sqlite3.Cursor:
        return self.get_connection(readonly=readonly).cursor()

    def get_connection(self, readonly=True) -> sqlite3.Connection:
        """
        Returns a connection to the database. By default, the returned connection is read-only
        (uri=True). Pass write=True to get a read-write connection. Only one read-write connection
        can exist at a time.
        """
        thread_id = threading.get_ident()
        with self._conn_dict_lock:
            conn_dict = self._conn_dict_readonly if readonly else self._conn_dict_readwrite
            conn = conn_dict.get(thread_id, None)
            logger.debug(f"get_connection: db_filename={self._db_filename} thread-id={thread_id}, readonly={readonly}, conn={conn}")
            if conn is None:
                conn = self._create_connection(readonly)
                conn_dict[thread_id] = conn
                if not readonly and len(conn_dict) > 1:
                    raise ValueError("Only one read-write connection is allowed (thread-id's: %s)" % list(conn_dict.keys()))
            return conn

    def _create_connection(self, readonly=True) -> sqlite3.Connection:
        """
        Assumes that self._conn_dict_lock is already acquired.
        """
        if not os.path.isfile(self._db_filename):
            if not self._create_cmds:
                raise ValueError(f"Database file {self._db_filename} does not exist and create_cmds is not specified.")

            create_conn = sqlite3.connect(self._db_filename)
            cursor = create_conn.cursor()
            for cmd in self._create_cmds:
                cursor.execute(cmd)
            create_conn.commit()
            create_conn.close()

        return sqlite3.connect(self._db_filename, uri=readonly)
