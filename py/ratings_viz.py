#!/usr/bin/env python3

"""
Use this script to visualize the evolution of ratings of an alphazero run.

Usage:

cd py;

./alphazero/main_loop.py -t <TAG>

./assign_ratings.py -t <TAG>

./ratings_viz.py -t <TAG>
"""
import argparse
import os
import pipes
import sqlite3
import sys
from collections import defaultdict, Counter

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure, curdoc

from config import Config


class Args:
    launch: bool
    alphazero_dir: str
    game: str
    tag: str
    port: int

    @staticmethod
    def load(args):
        Args.launch = bool(args.launch)
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tag = args.tag
        Args.port = args.port

        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    parser.add_argument('-p', '--port', type=int, default=5006, help='bokeh port (default: %(default)s)')

    args = parser.parse_args()
    Args.load(args)


load_args()

if not Args.launch:
    script = os.path.abspath(__file__)
    args = ' '.join(map(pipes.quote, sys.argv[1:] + ['--launch']))
    cmd = f'bokeh serve --port {Args.port} --show {script} --args {args}'
    sys.exit(os.system(cmd))


base_dir = os.path.join(Args.alphazero_dir, Args.game, Args.tag)


class ProgressVisualizer:
    def __init__(self, manager_path: str):
        self._conn = None
        self.manager_path = manager_path
        self.arena_dir = os.path.join(manager_path, 'arena')
        self.gating_logs_dir = os.path.join(manager_path, 'grading-logs')
        self.db_filename = os.path.join(self.arena_dir, 'arena.db')

        self.rand_agent_id = None
        self.perfect_agent_id = None
        self.iters_gen_agent_id_dict = defaultdict(dict)  # iters -> { gen -> rowid }

        self.gens = None
        self.match_dict = None
        self.load_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_filename)
        return self._conn

    def load_db(self):
        cursor = self.conn.cursor()

        agents = list(cursor.execute('SELECT rowid, rand, perfect, gen, iters FROM agents'))
        ratings = list(cursor.execute('SELECT agent_id, match_id, beta FROM ratings'))

        perfect_id = [a[0] for a in agents if a[2]][0]
        random_id = [a[0] for a in agents if a[1]][0]
        mcts_agents = [a for a in agents if not a[1] and not a[2]]
        iter_counts = Counter()
        for a in mcts_agents:
            iter_counts[a[4]] += 1
        most_common_iter_count = iter_counts.most_common(1)[0][0]

        mcts_id_to_gen = {a[0]: a[3] for a in mcts_agents if a[4] == most_common_iter_count}
        gens = np.array([g for i, g in mcts_id_to_gen.items()])

        match_id_to_ratings = defaultdict(dict)
        for r in ratings:
            match_id_to_ratings[r[1]][r[0]] = r[2]

        match_id_set = set(r[1] for r in ratings)
        match_ids = list(sorted(match_id_set))

        match_dict = {}
        for m in match_ids:
            perfect_rating = match_id_to_ratings[m][perfect_id]
            random_rating = match_id_to_ratings[m][random_id]
            mcts_ratings = np.array([match_id_to_ratings[m][i] for i in mcts_id_to_gen])
            match_dict[m] = (perfect_rating, random_rating, mcts_ratings)

        self.gens = gens
        self.match_dict = match_dict

    def plot(self):
        source = ColumnDataSource()

        match_ids = list(self.match_dict.keys())
        num_match_ids = len(self.match_dict)
        round_slider = Slider(title='Arena Round', start=1, end=num_match_ids, step=1, value=1)

        def update_helper(match_id):
            # TODO: for horizontal lines, use a Span, dynamically adjusting location using js_* method.
            # See: https://stackoverflow.com/a/76127040/543913
            (perfect_rating, random_rating, mcts_ratings) = self.match_dict[match_id]
            source.data = {
                'm': mcts_ratings,
                'p': [perfect_rating] * len(mcts_ratings),
                'r': [random_rating] * len(mcts_ratings),
                'g': self.gens,
            }

        def update(attr, old, new):
            r = round_slider.value
            match_id = match_ids[r - 1]
            update_helper(match_id)

        round_slider.on_change('value', update)
        update_helper(match_ids[0])

        plot = figure(height=600, width=800, title='Rating Graph',
                      y_axis_label='Rating', x_axis_label='Generation')
        plot.line('g', 'm', source=source, line_color='blue')
        plot.line('g', 'p', source=source, line_color='green')
        plot.line('g', 'r', source=source, line_color='red')

        inputs = column(plot, round_slider)
        return inputs


viz = ProgressVisualizer(base_dir)


curdoc().add_root(viz.plot())
