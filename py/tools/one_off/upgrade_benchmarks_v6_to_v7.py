#!/usr/bin/env python3
from alphazero.logic import constants
from alphazero.logic.run_params import RunParams
from util.py_util import CustomHelpFormatter
from util.sqlite3_util import DatabaseConnectionPool

import argparse
import shutil
import os
from pathlib import Path

"""
Upgrade benchmark directories from version 6 to version 7.

Key change in v7:
- The `mcts_agents` table in the benchmark database now includes a new column: `paradigm`.

Migration steps:
1. Identify benchmark directories to process:
   - If `--game` / `-g` is provided, process all directories under `/workspace/mount/v6/benchmarks/{game}/`.
   - If both `--game` and `--tag` are provided, process only `/workspace/mount/v6/benchmarks/{game}/{tag}/`.
   - Otherwise, process all game directories under `/workspace/mount/v6/benchmarks/`.

2. Copy all files into the corresponding v7 benchmark directory, excluding `{game}/{tag}/databases/benchmark.db`.

3. Create a new database in the v7 benchmark directory:
   - All tables are copied as-is, except for `mcts_agents`.
   - The `mcts_agents` table is recreated with the additional `paradigm` column.
   - All rows are migrated, with `paradigm` set to `'alpha0'` for every row.
"""

V6_BENCHMARK_DIR = Path('/workspace/mount/v6/benchmarks')
V7_BENCHMARK_DIR = Path('/workspace/mount/v7/benchmarks')


def copy_to_v7_dir(game: str, tag: str):
    v6_dir = V6_BENCHMARK_DIR / game / tag
    v7_dir = V7_BENCHMARK_DIR / game / tag
    if not v6_dir.exists():
        raise Exception(f"Source directory {v6_dir} does not exist.")

    for root, _, files in os.walk(v6_dir):
        rel_path = os.path.relpath(root, v6_dir)
        dest_dir = v7_dir / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if root == str(v6_dir / 'databases') and file == 'benchmark.db':
                continue
            src_file = Path(root) / file
            dest_file = dest_dir / file
            shutil.copy2(src_file, dest_file)


def create_v7_benchmark_db(game: str, tag: str):
    v6_db = V6_BENCHMARK_DIR / game / tag / 'databases' / 'benchmark.db'
    v7_db = V7_BENCHMARK_DIR / game / tag / 'databases' / 'benchmark.db'

    if v7_db.exists():
        print(f"v7 benchmark database {v7_db} already exists. Skipping creation.")
        return

    v7_db.parent.mkdir(parents=True, exist_ok=True)

    v6_db_conn_pool = DatabaseConnectionPool(v6_db)
    v6_conn = v6_db_conn_pool.get_connection()
    v6_cursor = v6_conn.cursor()

    v7_db_conn_pool = DatabaseConnectionPool(v7_db, constants.ARENA_TABLE_CREATE_CMDS)
    v7_conn = v7_db_conn_pool.get_connection()
    v7_cursor = v7_conn.cursor()

    # Copy all tables except mcts_agents
    v6_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='mcts_agents'")
    tables = v6_cursor.fetchall()
    for (table_name,) in tables:
        v6_cursor.execute(f"SELECT * FROM {table_name}")
        rows = v6_cursor.fetchall()
        if rows:
            placeholders = ','.join(['?'] * len(rows[0]))
            v7_cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)

    v6_cursor.execute("SELECT id, gen, n_iters, tag, is_zero_temp FROM mcts_agents")
    rows = v6_cursor.fetchall()
    for row in rows:
        id_, gen, n_iters, tag, is_zero_temp = row
        v7_cursor.execute("""
            INSERT INTO mcts_agents (id, paradigm, gen, n_iters, tag, is_zero_temp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id_, "alpha0", gen, n_iters, tag, is_zero_temp))

    v7_conn.commit()
    v6_conn.close()


def upgrade_benchmark(game: str, tag: str):
    if tag == 'reference.player':
        return

    try:
        copy_to_v7_dir(game, tag)
        create_v7_benchmark_db(game, tag)
        print(f"Successfully upgraded benchmark {game}/{tag} to v7.")
    except Exception as e:
        print(f"Failed to upgrade benchmark {game}/{tag} to v7: {e}")


def upgrade_game_benmarks(game: str):
    game_dir = V6_BENCHMARK_DIR / game
    if not game_dir.exists():
        raise Exception(f"Game directory {game_dir} does not exist.")
    for tag_dir in game_dir.iterdir():
        if tag_dir.is_dir():
            upgrade_benchmark(game, tag_dir.name)


def upgrade_all_benchmarks():
    for game_dir in V6_BENCHMARK_DIR.iterdir():
        if game_dir.is_dir():
            upgrade_game_benmarks(game_dir.name)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    return parser.parse_args()


def main():
    args = load_args()
    try:
        run_params = RunParams.create(args, require_tag=False)
    except Exception:
        run_params = None

    if run_params is None:
        upgrade_all_benchmarks()
    elif run_params.game and not run_params.tag:
        upgrade_game_benmarks(run_params.game)
    elif run_params.game and run_params.tag:
        upgrade_benchmark(run_params.game, run_params.tag)

if __name__ == "__main__":
    main()
