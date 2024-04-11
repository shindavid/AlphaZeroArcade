"""
Used by launch_dashboard.py to create a ratings plot.
"""
from alphazero.logic.custom_types import RatingTag
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
import games.index as game_index

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Span, RadioGroup, CheckboxGroup, Button
from bokeh.palettes import Category20
from bokeh.plotting import figure
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from collections import defaultdict
import os
import sqlite3
from typing import List


WIDTH = 800
HEIGHT = 600


def create_ratings_figure(output_dir: str, game: str, tags: List[str]):
    if not tags:
        # return a simple bokeh plot with a message
        return figure(title='Please select a tag', width=WIDTH, height=HEIGHT)
    data_list = get_rating_data_list(output_dir, game, tags)
    viz = ProgressVisualizer(game, data_list)
    return viz.root


class RatingData:
    def __init__(self, run_params: RunParams, rating_tag: RatingTag):
        organizer = DirectoryOrganizer(run_params)
        game_spec = game_index.get_game_spec(run_params.game)

        conn = sqlite3.connect(organizer.ratings_db_filename)
        cursor = conn.cursor()
        res = cursor.execute('SELECT mcts_gen, rating FROM ratings WHERE tag = ? '
                             'ORDER BY mcts_gen', (rating_tag,))
        gen_ratings = res.fetchall()
        conn.close()

        x_select_vars = ['timestamps.gen', 'positions_evaluated', 'batches_evaluated',
                         'games', 'end_timestamp - start_timestamp']
        x_select_var_str = ', '.join(x_select_vars)

        x_columns = ['mcts_gen', 'n_evaluated_positions', 'n_batches_evaluated', 'n_games', 'runtime']
        x_query = f'SELECT {x_select_var_str} FROM self_play_metadata JOIN timestamps USING (gen)'

        conn = sqlite3.connect(organizer.self_play_db_filename)
        cursor = conn.cursor()
        x_values = cursor.execute(x_query).fetchall()
        conn.close()

        gen_df = pd.DataFrame(gen_ratings, columns=['mcts_gen', 'rating']).set_index('mcts_gen')
        x_df = pd.DataFrame(x_values, columns=x_columns).set_index('mcts_gen')

        if len(x_df['runtime']) > 0:
            # Earlier versions stored runtimes in sec, not ns. This heuristic corrects the
            # earlier versions to ns.
            ts = max(x_df['runtime'])
            if ts > 1e9:
                x_df['runtime'] *= 1e-9  # ns -> sec

        window_length = 17
        y = gen_df['rating']
        if len(gen_df) > window_length:
            y2 = savgol_filter(y, window_length=window_length, polyorder=2)
            max_strength = game_spec.reference_player_family.max_strength
            gen_df['rating_smoothed'] = np.minimum(y2, max_strength)
        else:
            gen_df['rating_smoothed'] = y

        for col in x_df:
            x_df[col] = x_df[col].cumsum()

        gen_df = gen_df.join(x_df, how='inner').reset_index()

        tag = run_params.tag
        self.tag = tag
        self.rating_tag = rating_tag
        self.gen_df = gen_df
        self.label = tag
        if rating_tag:
            self.label = f'{tag}:{rating_tag}'


def make_rating_data_list(run_params: RunParams) -> List[RatingData]:
    organizer = DirectoryOrganizer(run_params)
    db_filename = organizer.ratings_db_filename
    if not os.path.exists(db_filename):
        return []

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    # find all distinct rating_tag's from ratings table:
    res = cursor.execute('SELECT DISTINCT tag FROM ratings')
    rating_tags = [r[0] for r in res.fetchall()]
    conn.close()

    return [RatingData(run_params, t) for t in rating_tags]


def get_rating_data_list(output_dir: str, game: str, tags: List[str]):
    if not tags:
        game_dir = os.path.join(output_dir, game)

        tags = os.listdir(game_dir)
        # sort tags by mtime:
        tags = sorted(tags, key=lambda t: os.stat(
            os.path.join(game_dir, t)).st_mtime)

    data_list = []
    for tag in tags:
        run_params = RunParams(output_dir=output_dir, game=game, tag=tag)
        data_list.extend(make_rating_data_list(run_params))

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

    def __init__(self, game: str, data_list: List[RatingData]):
        self.game = game
        self.y_variable = 'rating_smoothed'
        self.x_var_index = 0
        self.sources = defaultdict(ColumnDataSource)

        game = game_index.get_game_spec(game)
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
                self.min_x_dict[col] = mx if col not in self.min_x_dict else \
                    min(self.min_x_dict[col], mx)
                mx = max(x)
                self.max_x_dict[col] = mx if col not in self.max_x_dict else \
                    max(self.max_x_dict[col], mx)

            x = data[cls.X_VAR_COLUMNS[self.x_var_index]]
            y = data[self.y_variable]
            self.sources[rating_data.label].data = {'x': x, 'y': y}

            for y_var in ('rating', 'rating_smoothed'):
                my = max(data[y_var])
                self.max_y = my if self.max_y is None else max(self.max_y, my)

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

        title = f'{self.game} Alphazero Ratings'
        plot = figure(width=WIDTH, height=HEIGHT, title=title, x_range=x_range, y_range=y_range,
                      y_axis_label='Rating', x_axis_label=cls.X_VARS[self.x_var_index],
                      active_scroll='xwheel_zoom',
                      tools='pan,box_zoom,xwheel_zoom,reset,save')
        hline = Span(location=self.y_limit, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
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

        inputs = column(plot, row(column(checkbox_group, reload_button, realign_button),
                                  radio_group))
        return plot, inputs
