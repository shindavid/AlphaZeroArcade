from alphazero.logic.run_params import RunParams
from util.sqlite3_util import DatabaseConnectionPool

from pathlib import Path
from typing import Optional
import os
import shutil


class DatabaseTable:
    """
    Represents a single table being migrated or transformed from one SQLite database file to another.

    A `DatabaseTable` is constructed with the table's name and the source and destination database
    paths. It provides utility methods to copy data or to copy data while adding a new column with
    a constant value. These methods are intended to be scheduled by an upgrader as part of a schema
    migration between different output versions.
    """
    def __init__(self, name: str, src_db_path: Path, dest_db_path: Path):
        assert src_db_path.suffix == '.db', "DatabaseTable path must point to a .db file"
        assert dest_db_path.suffix == '.db', "DatabaseTable path must point to a .db file"
        self.name = name
        self.src_db_path = src_db_path
        self.dest_db_path = dest_db_path

    def schedule_column_add(self, new_column_name: str, new_column_value):
        """
        Copy all rows from src_db.table -> dest_db.table, adding a new constant column.
        Assumes dest table exists. Adds the column if missing.
        """
        table = self.name

        dest_pool = DatabaseConnectionPool(self.dest_db_path)
        dest_conn = dest_pool.get_connection()
        cur = dest_conn.cursor()

        # attach source db to the destination connection
        cur.execute("ATTACH DATABASE ? AS src", (str(self.src_db_path),))

        # source column order
        src_cols = [r[1] for r in cur.execute(f"PRAGMA src.table_info({table})")]
        if not src_cols:
            cur.execute("DETACH DATABASE src")
            return

        # ensure destination has the extra column
        dest_cols = [r[1] for r in cur.execute(f"PRAGMA main.table_info({table})")]
        if new_column_name not in dest_cols:
            raise Exception(f"Destination database {self.dest_db_path} table {table} is missing column {new_column_name}")

        src_csv = ", ".join(c for c in src_cols)
        dst_csv = src_csv + ", " + new_column_name

        dest_conn.execute("BEGIN")
        try:
            # copy rows and append the constant
            sql = (
                f'INSERT INTO {table} ({dst_csv}) '
                f'SELECT {src_csv}, ? FROM src.{table}'
            )
            cur.execute(sql, (new_column_value,))
            dest_conn.commit()
        except Exception:
            dest_conn.rollback()
            raise
        finally:
            cur.execute("DETACH DATABASE src")


    def schedule_copy(self):
        """
        Copy all rows from self.name in src_db_path into the same table in dest_db_path.
        Assumes schema is already identical in both DBs.
        """
        table = self.name


        dest_pool = DatabaseConnectionPool(self.dest_db_path)
        dest_conn = dest_pool.get_connection()
        cur = dest_conn.cursor()

        # Attach source database to this connection
        cur.execute("ATTACH DATABASE ? AS src", (str(self.src_db_path),))

        try:
            # Get column list from source table (preserve order)
            src_cols = [r[1] for r in cur.execute(f"PRAGMA src.table_info({table})")]
            if not src_cols:
                return  # nothing to copy

            cols_csv = ", ".join(c for c in src_cols)

            dest_conn.execute("BEGIN")
            try:
                # Copy rows
                sql = (
                    f'INSERT INTO {table} ({cols_csv}) '
                    f'SELECT {cols_csv} FROM src.{table}'
                )
                cur.execute(sql)
                dest_conn.commit()
            except Exception:
                dest_conn.rollback()
                raise
        finally:
            cur.execute("DETACH DATABASE src")


class UpgraderBase:
    """
    Base class for upgrading versioned output directories.

    The upgrader copies files from a source output directory tree(determined by `FROM_VERSION`)
    into a target output directory tree (determined by `TO_VERSION`). It traverses games and tags
    under the source directory and creates a mirrored structure in the target. By default, all
    files are copied verbatim; subclasses can override `get_creation_func` to provide custom
    creation logic for specific files (e.g., rebuilding databases with a modified schema).

    Typical usage is to subclass this base and set `FROM_VERSION` and`TO_VERSION` to concrete
    values. The subclass implements `get_creation_func(file)` to return a function that knows
    how to build the corresponding file in the target directory. If `None` is returned, the file
    is copied with `shutil.copy2`.

    Please refer to `upgrade_output_dirs_v7_to_v8.py` for an example subclass that adds a new
    column to specific tables in evaluation databases.
    """

    FROM_VERSION: int = -1
    TO_VERSION: int = -1

    def __init__(self, run_params: Optional[RunParams]):
        self._run_params = run_params

    def upgrade_all_output_dirs(self):
        for game_dir in self.from_output_dir().iterdir():
            if game_dir.is_dir():
                self.upgrade_game_output_dirs(game_dir.name)

    def upgrade_game_output_dirs(self, game: str):
        from_game_dir = self.from_output_dir() / game
        if not from_game_dir.exists():
            raise Exception(f"Source directory {from_game_dir} does not exist.")

        for tag_dir in from_game_dir.iterdir():
            if tag_dir.is_dir():
                self.upgrade_single_output_dir(game, tag_dir.name)

    def upgrade_single_output_dir(self, game: str, tag: str):
        from_dir = self.from_output_dir() / game / tag
        to_dir = self.to_output_dir() / game / tag

        if to_dir.exists():
            print(f"target output directory {self.to_output_dir() / game / tag} already exists. Skipping creation.")
            return

        if not from_dir.exists():
            raise Exception(f"Source directory {from_dir} does not exist.")

        print(f"Upgrading {game}/{tag} to v{self.TO_VERSION}...")

        for root, _, files in os.walk(from_dir):
            rel_path = os.path.relpath(root, from_dir)
            dest_dir = to_dir / rel_path
            dest_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                src_file_path = Path(root) / file
                dest_file_path = dest_dir / file

                creation_func = self.get_creation_func(src_file_path)
                if creation_func is not None:
                    creation_func(dest_file_path)
                else:
                    shutil.copy2(src_file_path, dest_file_path)

        print(f"Successfully upgraded {game}/{tag} to v{self.TO_VERSION}.")

    def get_creation_func(self, file: str) -> Optional[callable]:
        return None

    def run(self):
        if self._run_params is None:
            self.upgrade_all_output_dirs()
        elif self._run_params.game and not self._run_params.tag:
            self.upgrade_game_output_dirs(self._run_params.game)
        elif self._run_params.game and self._run_params.tag:
            self.upgrade_single_output_dir(self._run_params.game, self._run_params.tag)

    @classmethod
    def from_output_dir(cls) -> Path:
        return Path('/workspace/mount')/f'v{cls.FROM_VERSION}'/'output'

    @classmethod
    def to_output_dir(cls) -> Path:
        return Path('/workspace/mount')/f'v{cls.TO_VERSION}'/'output'
