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
Upgrade output directories from version 7 to version 8.

Key change in v8:
- The `matches` and 'evaluator_ratings` tables in the evaluation/ database now include a new column: `rating_tag`.

Migration steps:
1. Identify output directories in v7
    - If `--game` / `-g` is provided, process all directories under `/workspace/mount/v7/output/{game}/`.
    - If both `--game` and `--tag` are provided, process only `/workspace/mount/v7/output/{game}/{tag}/`.
    - Otherwise, process all game directories under `/workspace/mount/v7/output/`.

2. Copy all files into the corresponding v8 output directory, excluding `{game}/{tag}/databases/evaluation/{benchmark_tag}.db`.

3. Create new evaluation databases in the v8 output directory:
    - All tables are copied as-is, except for `matches` and `evaluator_ratings`.
    - The `matches` and `evaluator_ratings` tables are recreated with the additional `rating_tag` column.
    - All rows are migrated, with `rating_tag` set to 'default' for every row.
"""

V7_OUTPUT_DIR = Path('/workspace/mount/v7/output')
V8_OUTPUT_DIR = Path('/workspace/mount/v8/output')


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    return parser.parse_args()


def copy_to_v8_dir(game: str, tag: str):
    v7_dir = V7_OUTPUT_DIR / game / tag
    v8_dir = V8_OUTPUT_DIR / game / tag
    if not v7_dir.exists():
        raise Exception(f"Source directory {v7_dir} does not exist.")

    for root, _, files in os.walk(v7_dir):
        rel_path = os.path.relpath(root, v7_dir)
        dest_dir = v8_dir / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if Path(root).name == 'evaluation':
                continue
            src_file = Path(root) / file
            dest_file = dest_dir / file
            shutil.copy2(src_file, dest_file)


def create_v8_evaluation_db(game: str, tag: str, benchmark_tag: str):
    v7_db = V7_OUTPUT_DIR / game / tag / 'databases' / 'evaluation' / f'{benchmark_tag}.db'
    v8_db = V8_OUTPUT_DIR / game / tag / 'databases' / 'evaluation' / f'{benchmark_tag}.db'

    if v8_db.exists():
        print(f"v8 evaluation database {v8_db} already exists. Skipping creation.")
        return

    if not v7_db.exists():
        raise Exception(f"Source database {v7_db} does not exist.")

    v8_db.parent.mkdir(parents=True, exist_ok=True)

    v7_db_conn_pool = DatabaseConnectionPool(v7_db)
    v8_db_conn_pool = DatabaseConnectionPool(v8_db, constants.ARENA_TABLE_CREATE_CMDS)
    v7_conn = v7_db_conn_pool.get_connection()
    v8_conn = v8_db_conn_pool.get_connection()
    v7_cursor = v7_conn.cursor()
    v8_cursor = v8_conn.cursor()

    # Copy all tables except matches and evaluator_ratings
    v7_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('matches', 'evaluator_ratings')")
    tables = v7_cursor.fetchall()
    for (table_name,) in tables:
        v7_cursor.execute(f"SELECT * FROM {table_name}")
        rows = v7_cursor.fetchall()
        if rows:
            placeholders = ','.join(['?'] * len(rows[0]))
            v8_cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)

    # Copy over matches table with new schema
    v7_cursor.execute("SELECT id, agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type FROM matches")
    rows = v7_cursor.fetchall()
    for row in rows:
        id_, agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type_ = row
        v8_cursor.execute("""
            INSERT INTO matches (id, rating_tag, agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (id_, 'default', agent_id1, agent_id2, agent1_wins, agent2_wins, draws, type_))

    # Copy over evaluator_ratings table with new schema
    v7_cursor.execute("SELECT id, agent_id, rating FROM evaluator_ratings")
    rows = v7_cursor.fetchall()
    for row in rows:
        id_, agent_id, rating = row
        v8_cursor.execute("""
            INSERT INTO evaluator_ratings (id, rating_tag, agent_id, rating)
            VALUES (?, ?, ?, ?)
        """, (id_, 'default', agent_id, rating))

    v8_conn.commit()
    v8_conn.close()
    v7_conn.close()


def create_evaluation_dir(game: str, tag: str):
    v7_eval_dir = V7_OUTPUT_DIR / game / tag / 'databases' / 'evaluation'
    if not v7_eval_dir.exists():
        return

    for db_file in v7_eval_dir.iterdir():
        if db_file.suffix == '.db':
            benchmark_tag = db_file.stem
            create_v8_evaluation_db(game, tag, benchmark_tag)


def upgrade_single_output_dir(game: str, tag: str):
    if (V8_OUTPUT_DIR / game / tag).exists():
        print(f"v8 output directory {V8_OUTPUT_DIR / game / tag} already exists. Skipping creation.")
        return

    try:
        print(f"Upgrading {game}/{tag} to v8...")
        copy_to_v8_dir(game, tag)
        create_evaluation_dir(game, tag)
        print(f"Successfully upgraded {game}/{tag} to v8.")
    except Exception as e:
        print(f"Failed to upgrade {game}/{tag} to v8: {e}")


def upgrade_game_output_dirs(game: str):
    game_v7_dir = V7_OUTPUT_DIR / game
    if not game_v7_dir.exists():
        raise Exception(f"Source directory {game_v7_dir} does not exist.")

    for tag_dir in game_v7_dir.iterdir():
        if tag_dir.is_dir():
            upgrade_single_output_dir(game, tag_dir.name)


def upgrade_all_output_dirs():
    for game_dir in V7_OUTPUT_DIR.iterdir():
        if game_dir.is_dir():
            upgrade_game_output_dirs(game_dir.name)


def main():
    args = load_args()
    try:
        run_params = RunParams.create(args, require_tag=False)
    except Exception as e:
        run_params = None

    if run_params is None:
        upgrade_all_output_dirs()
    elif run_params.game and not run_params.tag:
        upgrade_game_output_dirs(run_params.game)
    elif run_params.game and run_params.tag:
        upgrade_single_output_dir(run_params.game, run_params.tag)


if __name__ == '__main__':
    main()
