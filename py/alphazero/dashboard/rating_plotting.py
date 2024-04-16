"""
Used by launch_dashboard.py to create a ratings plot.
"""
from .x_var_logic import XVarSelector, make_x_df
from alphazero.logic.custom_types import RatingTag
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
import games.index as game_index
from util import bokeh_util

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Span, CheckboxGroup
from bokeh.plotting import figure
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import os
import sqlite3
from typing import Dict, List


def create_ratings_figure(output_dir: str, game: str, tags: List[str]):
    if not tags:
        return figure(title='No data available')
    data_list = get_rating_data_list(output_dir, game, tags)
    plotter = RatingPlotter(game, data_list)
    return plotter.figure


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

        gen_df = pd.DataFrame(gen_ratings, columns=['mcts_gen', 'rating']).set_index('mcts_gen')

        window_length = 17
        y = gen_df['rating']
        if len(gen_df) > window_length:
            y2 = savgol_filter(y, window_length=window_length, polyorder=2)
            max_strength = game_spec.reference_player_family.max_strength
            gen_df['rating_smoothed'] = np.minimum(y2, max_strength)
        else:
            gen_df['rating_smoothed'] = y

        x_df = make_x_df(organizer)
        gen_df = gen_df.join(x_df, how='inner').reset_index()

        tag = run_params.tag
        self.tag = tag
        self.rating_tag = rating_tag
        self.gen_df = gen_df
        self.label = tag
        if rating_tag:
            self.label = f'{tag}:{rating_tag}'

        self.valid = len(gen_df) > 0


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

    data_list = [RatingData(run_params, t) for t in rating_tags]
    return [rd for rd in data_list if rd.valid]


def get_rating_data_list(output_dir: str, game: str, tags: List[str]):
    data_list = []
    for tag in tags:
        run_params = RunParams(output_dir=output_dir, game=game, tag=tag)
        data_list.extend(make_rating_data_list(run_params))

    return data_list


class RatingPlotter:
    def __init__(self, game: str, data_list: List[RatingData]):
        self.game = game

        self.x_var_selector = XVarSelector([rd.gen_df for rd in data_list])
        self.y_variable = 'rating_smoothed'
        self.sources: Dict[str, ColumnDataSource] = {}

        game = game_index.get_game_spec(game)
        self.y_limit = game.reference_player_family.max_strength

        self.plotted_labels = set()
        self.max_y = 0
        self.load(data_list)
        self.figure = self.make_figure()

    def load(self, data_list: List[RatingData]):
        self.data_list = data_list
        for rating_data in data_list:
            data = rating_data.gen_df

            source = ColumnDataSource(data)
            source.data['y'] = data[self.y_variable]
            self.x_var_selector.init_source(source)
            self.sources[rating_data.label] = source

            for y_var in ('rating', 'rating_smoothed'):
                self.max_y = max(self.max_y, max(data[y_var]))

    def add_lines(self, plot):
        data_list = self.data_list
        n = len(data_list)
        colors = bokeh_util.get_colors(n)
        for rating_data, color in zip(data_list, colors):
            label = rating_data.label
            if label in self.plotted_labels:
                continue
            self.plotted_labels.add(label)
            source = self.sources[label]
            plot.line('x', 'y', source=source, line_color=color, legend_label=label)

    def make_figure(self):
        y_range = [0, self.max_y+1]
        title = f'{self.game} Alphazero Ratings'
        plot = figure(title=title, x_range=[0, 1], y_range=y_range, y_axis_label='Rating',
                      active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')

        self.x_var_selector.init_plot(plot)

        hline = Span(location=self.y_limit, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)

        self.add_lines(plot)

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'

        checkbox_group = self.make_checkbox_group()
        radio_group = self.x_var_selector.create_radio_group([plot], list(self.sources.values()))

        return column(plot, row(checkbox_group, radio_group))

    def make_checkbox_group(self):
        checkbox_group = CheckboxGroup(labels=['Smoothed'], active=[0])

        def update_data(attr, old, new):
            smoothed = 0 in checkbox_group.active
            self.y_variable = 'rating_smoothed' if smoothed else 'rating'
            y_var_column = self.y_variable

            for rating_data in self.data_list:
                source = self.sources[rating_data.label]
                source.data['y'] = rating_data.gen_df[y_var_column]

        checkbox_group.on_change('active', update_data)

        return checkbox_group
