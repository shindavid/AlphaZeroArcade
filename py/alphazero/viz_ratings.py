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
from typing import List, Optional

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Span, RadioGroup, CheckboxGroup, Button
from bokeh.palettes import Category20
from bokeh.plotting import figure, curdoc

import games
from config import Config
from util.py_util import timed_print

from scipy.signal import savgol_filter


class Args:
    launch: bool
    alphazero_dir: str
    game: str
    tags: Optional[List[str]]
    mcts_iters_list: List[int]
    port: int

    @staticmethod
    def load(args):
        Args.launch = bool(args.launch)
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tags = [t for t in args.tag.split(',') if t] if args.tag else None
        Args.mcts_iters_list = [] if not args.mcts_iters else [int(s) for s in args.mcts_iters.split(',')]
        Args.tag = args.tag
        Args.port = args.port


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-t', '--tag', help='tag(s) for this run, comma-separated (e.g. "v1,v2"). If not specified, plots all tags')
    parser.add_argument('-i', '--mcts-iters', help='mcts-iters values to include (default: all), comma-separated (e.g. "300,3000")')
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

        self.tag = tag
        self.mcts_iters = mcts_iters
        self.gen_df = gen_df
        self.label = f'{tag} (i={mcts_iters})'


def make_rating_data_list(tag: Optional[str]=None) -> List[RatingData]:
    game_dir = os.path.join(Args.alphazero_dir, Args.game)
    if not tag:
        tags = os.listdir(game_dir)
        # sort tags by mtime:
        tags = sorted(tags, key=lambda t: os.stat(os.path.join(game_dir, t)).st_mtime)
        data_list = []
        for tag in tags:
            data_list.extend(make_rating_data_list(tag))
        return data_list

    base_dir = os.path.join(game_dir, tag)

    db_filename = os.path.join(base_dir, 'ratings.db')
    if not os.path.exists(db_filename):
        return []
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    # find all distinct mcts_iters values from ratings table:
    res = cursor.execute('SELECT DISTINCT mcts_iters FROM ratings')
    mcts_iters_list = [r[0] for r in res.fetchall()]
    conn.close()

    if Args.mcts_iters_list:
        mcts_iters_list = [m for m in mcts_iters_list if m in Args.mcts_iters_list]

    return [RatingData(tag, m) for m in mcts_iters_list]


def get_rating_data_list():
    data_list = []
    if Args.tags:
        for tag in Args.tags:
            data_list.extend(make_rating_data_list(tag))
    else:
        data_list = make_rating_data_list()
    return data_list


class ProgressVisualizer:

    X_VAR_DICT = {
        'Generation': 'mcts_gen',
        'Games': 'n_games',
        'Self-Play Runtime (sec)': 'runtime',
        'Num Evaluated Positions': 'n_evaluated_positions',
        'Num Evaluated Batches': 'n_batches_evaluated',
    }

    X_VARS = list(X_VAR_DICT.keys())
    X_VAR_COLUMNS = list(X_VAR_DICT.values())

    def __init__(self, data_list: List[RatingData]):
        self.y_variable = 'rating_smoothed'
        self.x_var_index = 0
        self.sources = defaultdict(ColumnDataSource)

        game = games.get_game_type(Args.game)
        self.y_limit = game.reference_player_family.max_strength

        self.plotted_labels = set()
        self.min_x_dict = {}
        self.max_x_dict = {}
        self.max_y = None
        self.load(data_list)
        self.plot, self.root = self.make_plot_and_root()

    def load(self, data_list: List[RatingData]):
        cls = ProgressVisualizer
        self.data_list = data_list
        for rating_data in data_list:
            data = rating_data.gen_df
            if not len(data):
                continue

            for col in data:
                x = data[col]
                mx = min(x)
                self.min_x_dict[col] = mx if col not in self.min_x_dict else min(self.min_x_dict[col], mx)
                mx = max(x)
                self.max_x_dict[col] = mx if col not in self.max_x_dict else max(self.max_x_dict[col], mx)

            x = data[cls.X_VAR_COLUMNS[self.x_var_index]]
            y = data[self.y_variable]

            my = max(y)
            self.max_y = my if self.max_y is None else max(self.max_y, my)
            self.sources[rating_data.label].data = { 'x': x, 'y': y }

    def reload_data(self):
        data_list = get_rating_data_list()
        self.load(data_list)
        self.add_lines(self.plot)

    def realign_plot(self):
        cls = ProgressVisualizer
        col = cls.X_VAR_COLUMNS[self.x_var_index]
        x_range = [self.min_x_dict[col], self.max_x_dict[col]]
        y_range = [0, self.max_y+1]
        self.plot.x_range.start = x_range[0]
        self.plot.x_range.end = x_range[1]
        self.plot.y_range.start = y_range[0]
        self.plot.y_range.end = y_range[1]

    def add_lines(self, plot):
        cls = ProgressVisualizer
        data_list = self.data_list
        n = len(data_list)
        if n <= 2:
            colors = Category20[3][:n]
        else:
            colors = Category20[n]
        for rating_data, color in zip(data_list, colors):
            label = rating_data.label
            if label in self.plotted_labels:
                continue
            self.plotted_labels.add(label)
            source = self.sources[label]
            source.data['x'] = rating_data.gen_df[cls.X_VAR_COLUMNS[self.x_var_index]]
            plot.line('x', 'y', source=source, line_color=color, legend_label=label)

    def make_plot_and_root(self):
        cls = ProgressVisualizer

        radio_group = RadioGroup(labels=cls.X_VARS, active=self.x_var_index)
        checkbox_group = CheckboxGroup(labels=['Smoothed'], active=[0])
        reload_button = Button(label='Reload data', button_type='primary')
        reload_button.on_click(self.reload_data)
        realign_button = Button(label='Realign plot', button_type='success')
        realign_button.on_click(self.realign_plot)

        if self.max_y is None:
            x_range = [0, 1]
            y_range = [0, 1]
        else:
            col = cls.X_VAR_COLUMNS[self.x_var_index]
            x_range = [self.min_x_dict[col], self.max_x_dict[col]]
            y_range = [0, self.max_y+1]

        title = f'{Args.game} Alphazero Ratings'
        plot = figure(height=600, width=800, title=title, x_range=x_range, y_range=y_range,
                      y_axis_label='Rating', x_axis_label=cls.X_VARS[self.x_var_index],
                      active_scroll='xwheel_zoom',
                      tools='pan,box_zoom,xwheel_zoom,reset,save')
        hline = Span(location=self.y_limit, dimension='width', line_color='gray', line_dash='dashed', line_width=1)
        plot.add_layout(hline)

        self.add_lines(plot)

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'

        def update_data(attr, old, new):
            prev_x_var_index = self.x_var_index
            self.x_var_index = radio_group.active
            smoothed = 0 in checkbox_group.active
            self.y_variable = 'rating_smoothed' if smoothed else 'rating'
            x_var_column = cls.X_VAR_COLUMNS[self.x_var_index]
            y_var_column = self.y_variable

            for rating_data in self.data_list:
                source = self.sources[rating_data.label]
                source.data['x'] = rating_data.gen_df[x_var_column]
                source.data['y'] = rating_data.gen_df[y_var_column]

            plot.xaxis.axis_label = cls.X_VARS[self.x_var_index]

            if self.x_var_index != prev_x_var_index:
                prev_x_var_column = cls.X_VAR_COLUMNS[prev_x_var_index]
                start = plot.x_range.start
                end = plot.x_range.end
                prev_x_min = self.min_x_dict[prev_x_var_column]
                prev_x_max = self.max_x_dict[prev_x_var_column]
                prev_x_width = prev_x_max - prev_x_min
                if prev_x_width > 0:
                    start_pct = (start - prev_x_min) / prev_x_width
                    end_pct = (end - prev_x_min) / prev_x_width
                    x_min = self.min_x_dict[x_var_column]
                    x_max = self.max_x_dict[x_var_column]
                    x_width = x_max - x_min
                    plot.x_range.start = x_min + start_pct * x_width
                    plot.x_range.end = x_min + end_pct * x_width

        widgets = [radio_group, checkbox_group]
        for widget in widgets:
            widget.on_change('active', update_data)

        inputs = column(plot, row(column(checkbox_group, reload_button, realign_button), radio_group))
        return plot, inputs


def main():
    load_args()

    if not Args.launch:
        script = os.path.abspath(__file__)
        args = ' '.join(map(pipes.quote, sys.argv[1:] + ['--launch']))
        cmd = f'bokeh serve --port {Args.port} --show {script} --args {args}'
        sys.exit(os.system(cmd))
    else:
        data_list = get_rating_data_list()
        viz = ProgressVisualizer(data_list)

        curdoc().title = Args.game
        curdoc().add_root(viz.root)


main()
