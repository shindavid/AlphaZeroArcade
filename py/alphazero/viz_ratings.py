#!/usr/bin/env python3

"""
Use this script to visualize the progress of one or more alphazero runs.

Usage:

cd py;

./alphazero/main_loop.py -g <GAME> -t <TAG>

While the above is running, launch the following, preferably from a different machine:

./alphazero/compute_ratings.py -g <GAME> -t <TAG> -D

Alternatively, if you don't have enough hardware, stop the main loop before running the above. You don't need daemon
mode (-D) in this case.

While the above is running, launch the visualizer:

./alphazero/viz_ratings.py -g <GAME> -t <TAG>

If you have multiple main-loops (running or completed), you can pass a comma-separated list of tags for the -t option
for both compute_ratings.py and viz_ratings.py.

The visualizer will show a graph based on the rating data generated so-far by compute_ratings.py. Refreshing the
browser window will cause the visualizer to update the graph with the latest data.
"""
import argparse
import os
import pipes
import sqlite3
import sys
from collections import defaultdict
from typing import List

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Span, RadioGroup, CheckboxGroup
from bokeh.palettes import Category10
from bokeh.plotting import figure, curdoc

import games
from config import Config
from util.py_util import timed_print

from scipy.signal import savgol_filter


class Args:
    launch: bool
    alphazero_dir: str
    game: str
    tags: List[str]
    port: int

    @staticmethod
    def load(args):
        assert args.tag, 'Required option: -t'
        Args.launch = bool(args.launch)
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tags = [t for t in args.tag.split(',') if t]
        Args.tag = args.tag
        Args.port = args.port


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-t', '--tag', help='tag(s) for this run, comma-separated (e.g. "v1,v2")')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    parser.add_argument('-p', '--port', type=int, default=5006, help='bokeh port (default: %(default)s)')

    args = parser.parse_args()
    Args.load(args)


class RatingData:
    def __init__(self, tag: str, mcts_iters: int):
        base_dir = os.path.join(Args.alphazero_dir, Args.game, tag)

        db_filename = os.path.join(base_dir, 'ratings.db')
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()

        res = cursor.execute('SELECT mcts_gen, rating FROM ratings WHERE mcts_iters = ? ORDER BY mcts_gen',
                             (mcts_iters,))

        gen_ratings = res.fetchall()

        x_values_columns = ['mcts_gen', 'n_games', 'runtime', 'n_evaluated_positions', 'n_batches_evaluated']
        res = cursor.execute('SELECT %s FROM x_values ORDER BY mcts_gen' % (', '.join(x_values_columns)))
        x_values = res.fetchall()

        gen_df = pd.DataFrame(gen_ratings, columns=['mcts_gen', 'rating']).set_index('mcts_gen')
        x_df = pd.DataFrame(x_values, columns=x_values_columns).set_index('mcts_gen')

        window_length = 17
        y = gen_df['rating']
        if len(gen_df) > window_length:
            y2 = savgol_filter(y, window_length=window_length, polyorder=2)
            gen_df['rating_smoothed'] = y2
        else:
            gen_df['rating_smoothed'] = y

        for col in x_df:
            x_df[col] = x_df[col].cumsum()

        assert set(gen_df.index).issubset(set(x_df.index))
        gen_df = gen_df.join(x_df).reset_index()

        conn.close()
        n = len(gen_df)
        timed_print(f'Loaded {n} rows of data from {db_filename}')

        self.tag = tag
        self.mcts_iters = mcts_iters
        self.gen_df = gen_df
        self.label = f'{tag} (i={mcts_iters})'


def make_rating_data_list(tag: str) -> List[RatingData]:
    base_dir = os.path.join(Args.alphazero_dir, Args.game, tag)

    db_filename = os.path.join(base_dir, 'ratings.db')
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    # find all distinct mcts_iters values from ratings table:
    res = cursor.execute('SELECT DISTINCT mcts_iters FROM ratings')
    mcts_iters_list = [r[0] for r in res.fetchall()]
    return [RatingData(tag, mcts_iters) for mcts_iters in mcts_iters_list]


class ProgressVisualizer:
    def __init__(self, data_list: List[RatingData]):
        self.data_list = data_list
        self.sources = defaultdict(ColumnDataSource)

        game = games.get_game_type(Args.game)
        self.y_limit = game.reference_player_family.max_strength

        self.max_x_dict = {}
        self.max_y = None
        for rating_data in data_list:
            data = rating_data.gen_df
            if not len(data):
                continue

            for col in data:
                x = data[col]
                mx = max(x)
                self.max_x_dict[col] = mx if col not in self.max_x_dict else max(self.max_x_dict[col], mx)

            y = data['rating_smoothed']

            my = max(y)
            self.max_y = my if self.max_y is None else max(self.max_y, my)
            self.sources[rating_data.label].data = { 'y': y }

    def plot(self):
        x_var_dict = {
            'Generation': 'mcts_gen',
            'Games': 'n_games',
            'Self-Play Runtime (sec)': 'runtime',
            'Num Evaluated Positions': 'n_evaluated_positions',
            'Num Evaluated Batches': 'n_batches_evaluated',
        }
        x_vars = list(x_var_dict.keys())
        x_var_columns = list(x_var_dict.values())
        default_x_var_index = 0
        default_x_var = x_vars[default_x_var_index]
        default_x_var_column = x_var_columns[default_x_var_index]

        radio_group = RadioGroup(labels=x_vars, active=default_x_var_index)
        checkbox_group = CheckboxGroup(labels=['Smoothed'], active=[0])

        data_list = self.data_list
        if self.max_y is None:
            x_range = [0, 1]
            y_range = [0, 1]
        else:
            x_range = [0, self.max_x_dict[default_x_var_column]]
            y_range = [0, self.max_y+1]

        title = f'{Args.game} Alphazero Ratings'
        plot = figure(height=600, width=800, title=title, x_range=x_range, y_range=y_range,
                      y_axis_label='Rating', x_axis_label=default_x_var,
                      active_scroll='xwheel_zoom',
                      tools='pan,box_zoom,xwheel_zoom,reset,save')
        hline = Span(location=self.y_limit, dimension='width', line_color='gray', line_dash='dashed', line_width=1)
        plot.add_layout(hline)

        n = len(data_list)
        if n <= 2:
            colors = Category10[3][:n]
        else:
            colors = Category10[n]
        for rating_data, color in zip(data_list, colors):
            source = self.sources[rating_data.label]
            source.data['x'] = rating_data.gen_df[default_x_var_column]
            label = rating_data.label
            plot.line('x', 'y', source=source, line_color=color, legend_label=label)

        plot.legend.location = 'bottom_right'

        def update_data(attr, old, new):
            x_var_index = radio_group.active
            smoothed = 0 in checkbox_group.active
            x_var_column = x_var_columns[x_var_index]
            y_var_column = 'rating_smoothed' if smoothed else 'rating'

            for rating_data in self.data_list:
                source = self.sources[rating_data.label]
                source.data['x'] = rating_data.gen_df[x_var_column]
                source.data['y'] = rating_data.gen_df[y_var_column]

            plot.x_range.end = self.max_x_dict[x_var_column]
            plot.xaxis.axis_label = x_vars[x_var_index]

        widgets = [radio_group, checkbox_group]
        for widget in widgets:
            widget.on_change('active', update_data)

        inputs = column(plot, row(checkbox_group, radio_group))
        return inputs


def main():
    load_args()

    if not Args.launch:
        script = os.path.abspath(__file__)
        args = ' '.join(map(pipes.quote, sys.argv[1:] + ['--launch']))
        cmd = f'bokeh serve --port {Args.port} --show {script} --args {args}'
        sys.exit(os.system(cmd))
    else:
        data_list = []
        for tag in Args.tags:
            data_list.extend(make_rating_data_list(tag))
        viz = ProgressVisualizer(data_list)

        curdoc().title = Args.game
        curdoc().add_root(viz.plot())


main()
