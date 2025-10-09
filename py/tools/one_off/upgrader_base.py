from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import VERSION
from util.py_util import CustomHelpFormatter
from util.sqlite3_util import escape_value

import argparse
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Optional


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    return parser.parse_args()


class DatabaseTable:
    def __init__(self, name: str, db_file: Path):
        self.name = name
        self.db_file = db_file
        self.pending_column_adds = []  # List of (column_name, column_value)

    def is_evaluation_db(self) -> bool:
        """
        Whether filename looks like ".../databases/evaluation/*.db"
        """
        parts = self.db_file.parts
        if len(parts) < 3:
            return False
        return parts[-3] == 'databases' and parts[-2] == 'evaluation'

    def schedule_column_add(self, new_column_name: str, new_column_value):
        """
        Schedule adding a new column with a constant value to all rows in this table.
        The actual operation will be performed later by the upgrader.
        """

        self.pending_column_adds.append((new_column_name, escape_value(new_column_value)))


class UpgraderBase:
    """
    Base class for upgrading versioned output directories.

    The upgrader copies files from a source output directory tree(determined by `FROM_VERSION`)
    into a target output directory tree (determined by `TO_VERSION`). It traverses games and tags
    under the source directory and creates a mirrored structure in the target. By default, all
    files are copied verbatim; subclasses can override `rewrite_db_table()` to provide custom
    logic for changing the schema or contents of specific database files.

    Typical usage is to subclass this base and set `FROM_VERSION` and`TO_VERSION` to concrete
    values, with an override of `rewrite_db_table()`.

    Please refer to `upgrade_output_dirs_v7_to_v8.py` for an example subclass that adds a new
    column to specific tables in evaluation databases.
    """

    FROM_VERSION: int = -1
    TO_VERSION: int = -1

    def __init__(self, run_params: Optional[RunParams]=None):
        if run_params is None:
            args = load_args()
            try:
                run_params = RunParams.create(args, require_tag=False)
            except Exception as e:
                pass

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

        error_list = []
        for root, _, files in os.walk(from_dir):
            rel_path = os.path.relpath(root, from_dir)
            dest_dir = to_dir / rel_path
            dest_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                src_file_path = Path(root) / file
                dest_file_path = dest_dir / file

                try:
                    shutil.copy2(src_file_path, dest_file_path)
                    extension = os.path.splitext(file)[1].lower()
                    if extension == '.db':
                        self.upgrade_db_file(dest_file_path)
                except Exception as e:
                    error_list.append((src_file_path, str(e)))

        if error_list:
            print("The following errors were encountered during upgrade:")
            for file_path, err_msg in error_list:
                print(f"  {file_path}: {err_msg}")
        else:
            print(f"Upgrade of {game}/{tag} to v{self.TO_VERSION} completed successfully.")

    def upgrade_db_file(self, db_file: Path):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]

        cmds = []
        for table_name in tables:
            db_table = DatabaseTable(table_name, db_file)
            self.rewrite_db_table(db_table)
            if db_table.pending_column_adds:
                for name, value in db_table.pending_column_adds:
                    cmd = f"ALTER TABLE {table_name} ADD COLUMN {name} DEFAULT {value}"
                    cmds.append(cmd)

        if not cmds:
            return

        print(f"Upgrading database {db_file}:")
        for cmd in cmds:
            print(f"  {cmd}")
            cur.execute(cmd)
        conn.commit()
        conn.close()

    def rewrite_db_table(self, table: DatabaseTable):
        """
        Override this method in subclasses to provide custom logic for rewriting specific
        database tables during the upgrade process. The default implementation does nothing,
        meaning that all database tables are copied verbatim.
        """
        pass

    def run(self):
        if self._run_params is None:
            self.upgrade_all_output_dirs()
        elif self._run_params.game and not self._run_params.tag:
            self.upgrade_game_output_dirs(self._run_params.game)
        elif self._run_params.game and self._run_params.tag:
            self.upgrade_single_output_dir(self._run_params.game, self._run_params.tag)

    @classmethod
    def from_output_dir(cls) -> Path:
        assert cls.FROM_VERSION == VERSION.num - 1
        return Path('/workspace/mount')/f'v{cls.FROM_VERSION}'/'output'

    @classmethod
    def to_output_dir(cls) -> Path:
        assert cls.TO_VERSION == VERSION.num
        return Path('/workspace/mount')/f'v{cls.TO_VERSION}'/'output'
