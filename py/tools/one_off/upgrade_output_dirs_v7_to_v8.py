#!/usr/bin/env python3
from alphazero.logic import constants
from alphazero.logic.run_params import RunParams
from tools.one_off.UpgraderBase import DatabaseTable, UpgraderBase
from util.py_util import CustomHelpFormatter
from util.sqlite3_util import DatabaseConnectionPool

import argparse
from pathlib import Path
"""
Upgrade output directories from version 7 to version 8.

Key change in v8:
- The `matches` and 'evaluator_ratings` tables in the evaluation/ database now include a new column: `rating_tag`.
"""


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    return parser.parse_args()


class Upgrader(UpgraderBase):
    FROM_VERSION = 7
    TO_VERSION = 8

    def get_creation_func(self, src_file_path: Path):
        if src_file_path.suffix == '.db' and src_file_path.parent.name == 'evaluation':
            def creation_func(dest_file_path: Path):
                src_db_conn_pool = DatabaseConnectionPool(src_file_path)
                src_conn = src_db_conn_pool.get_connection()
                src_cursor = src_conn.cursor()

                dest_db_conn_pool = DatabaseConnectionPool(dest_file_path, constants.ARENA_TABLE_CREATE_CMDS)
                dest_db_conn_pool.get_connection()

                src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = src_cursor.fetchall()
                for (table_name,) in tables:
                    db_table = DatabaseTable(table_name, src_file_path, dest_file_path)
                    if table_name in ('matches', 'evaluator_ratings'):
                        db_table.schedule_column_add('rating_tag', '')
                    else:
                        db_table.schedule_copy()

            return creation_func
        return None


def main():
    args = load_args()
    try:
        run_params = RunParams.create(args, require_tag=False)
    except Exception as e:
        run_params = None

    upgrader = Upgrader(run_params)
    upgrader.run()


if __name__ == '__main__':
    main()
