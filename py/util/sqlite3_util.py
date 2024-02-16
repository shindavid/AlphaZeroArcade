import os
import sqlite3
import threading
from typing import Dict, List, Optional


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
        self._conn_dict: Dict[int, sqlite3.Connection] = {}  # thread-id -> connection
        self._conn_dict_lock = threading.Lock()

    def get_cursor(self) -> sqlite3.Cursor:
        return self.get_connection().cursor()

    def get_connection(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        with self._conn_dict_lock:
            conn = self._conn_dict.get(thread_id, None)
            if conn is None:
                conn = self._create_connection()
                self._conn_dict[thread_id] = conn
            return conn

    def _create_connection(self) -> sqlite3.Connection:
        """
        Assumes that self._conn_dict_lock is already acquired.
        """
        if os.path.isfile(self._db_filename):
            return sqlite3.connect(self._db_filename)

        if not self._create_cmds:
            raise ValueError(f"Database file {self._db_filename} does not exist and create_cmds is not specified.")

        conn = sqlite3.connect(self._db_filename)
        c = conn.cursor()
        for cmd in self._create_cmds:
            c.execute(cmd)
        conn.commit()
        return conn
