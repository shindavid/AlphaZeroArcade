#!/usr/bin/env python3

"""
Use this script to visualize the progress of an othello alphazero run.

Usage:

cd py;

./alphazero/main_loop.py -g othello -t <TAG>

While the above is running, launch the grading daemon, preferably from a different machine:

./othello/assign_edax_ratings.py -t <TAG> -D

Alternatively, if you don't have enough hardware, stop the main loop before running the above. You don't need daemon
mode (-D) in this case.

While the above is running, launch the visualizer:

./othello/viz_edax_ratings.py -t <TAG>

The visualizer will show a graph based on the rating data generated so-far by assign_edax_ratings.py. Refreshing the
browser window will cause the visualizer to update the graph with the latest data.

If you have multiple tagged runs and wish to compare them on one plot, you can specify multiple tags, comma-separated:

./othello/viz_edax_ratings.py -t <TAG1>,<TAG2>,<TAG3>
"""
import argparse
import json
import os
import pipes
import sqlite3
import sys

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from bokeh.plotting import figure, curdoc

from config import Config
from util.py_util import timed_print


class Args:
    launch: bool
    alphazero_dir: str
    tag: str
    port: int

    @staticmethod
    def load(args):
        Args.launch = bool(args.launch)
        Args.alphazero_dir = args.alphazero_dir
        Args.tag = args.tag
        Args.port = args.port

        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-t', '--tag', help='tag(s) for this run, comma-separated (e.g. "v1,v2")')
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


class RatingData:
    def __init__(self, tag: str):
        base_dir = os.path.join(Args.alphazero_dir, 'othello', tag)

        metadata_filename = os.path.join(base_dir, 'metadata.json')
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        n_games = metadata['n_games']
        mcts_iters = metadata['mcts_iters']

        db_filename = os.path.join(base_dir, 'edax.db')
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        res = cursor.execute('SELECT mcts_gen, edax_rating FROM ratings WHERE mcts_iters = ? AND n_games >= ?',
                             (mcts_iters, n_games))

        gen_rating_pairs = []
        for mcts_gen, edax_rating in res.fetchall():
            gen_rating_pairs.append((mcts_gen, edax_rating))

        conn.close()
        n = len(gen_rating_pairs)
        timed_print(f'Loaded {n} rows of data from {db_filename}')

        gen_rating_pairs.sort()

        self.tag = tag
        self.n_games = n_games
        self.mcts_iters = mcts_iters
        self.gen_rating_pairs = gen_rating_pairs
        self.label = f'{tag} (i={mcts_iters}, G={n_games})'


data_list = [RatingData(tag) for tag in Args.tag.split(',')]


class ProgressVisualizer:
    def __init__(self):
        self.source = ColumnDataSource()
        self.data = {}

        self.max_x = None
        self.max_y = None

        for data in data_list:
            x = np.array([g[0] for g in data.gen_rating_pairs])
            y = np.array([g[1] for g in data.gen_rating_pairs])
            self.data[data.tag + '.x'] = x
            self.data[data.tag + '.y'] = y

            mx = max(x)
            my = max(y)
            self.max_x = mx if self.max_x is None else max(self.max_x, mx)
            self.max_y = my if self.max_y is None else max(self.max_y, my)

        self.source.data = self.data

    def plot(self):
        source = self.source
        x_range = [1, self.max_x]
        y_range = [0, self.max_y+1]

        title = f'Othello Alphazero Run'
        plot = figure(height=600, width=800, title=title, x_range=x_range, y_range=y_range,
                      y_axis_label='Edax Rating', x_axis_label='Generation')  # , tools='wheel_zoom')

        n = len(data_list)
        colors = viridis(n)
        for data, color in zip(data_list, colors):
            x = data.tag + '.x'
            y = data.tag + '.y'
            label = data.label
            plot.line(x, y, source=source, line_color=color, legend_label=label)

        plot.legend.location = 'top_left'
        inputs = column(plot)
        return inputs


viz = ProgressVisualizer()


curdoc().add_root(viz.plot())
