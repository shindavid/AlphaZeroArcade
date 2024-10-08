"""
Used by launch_dashboard.py to create a self-play plot.
"""
from .x_var_logic import XVarSelector, make_x_df
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Legend, LegendItem, Span
from bokeh.plotting import figure
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from typing import Dict, List


def create_self_play_figure(game: str, tags: List[str]):
    data_list: List[SelfPlayData] = []
    for tag in tags:
        run_params = RunParams(game, tag)
        organizer = DirectoryOrganizer(run_params)

        conn = sqlite3.connect(organizer.self_play_db_filename)
        data = SelfPlayData(conn, tag)
        conn.close()
        if not data.valid:
            continue

        x_df = make_x_df(organizer)
        data.join(x_df)
        data_list.append(data)

    if not data_list:
        return figure(title='No data available')

    plotter = SelfPlayPlotter(data_list)
    return plotter.figure


@dataclass
class SelectVar:
    table: str
    name: str
    sql_select_str: str


class SelfPlayData:
    SELECT_VARS = [
        SelectVar('metrics', 'mcts_gen', 'gen'),
        SelectVar('metrics', 'cache_hits', 'SUM(cache_hits)'),
        SelectVar('metrics', 'cache_misses', 'SUM(cache_misses)'),
        SelectVar('self_play_metadata', 'mcts_gen', 'gen'),
        SelectVar('self_play_metadata', 'positions_evaluated', 'positions_evaluated'),
        SelectVar('self_play_metadata', 'batches_evaluated', 'batches_evaluated'),
        SelectVar('self_play_metadata', 'y_runtime', 'runtime'),
        SelectVar('self_play_metadata', 'n_positions', 'augmented_positions'),
        SelectVar('self_play_metadata', 'y_games', 'games'),
    ]

    def __init__(self, conn: sqlite3.Connection, tag: str):
        self.tag = tag
        select_vars_by_table = defaultdict(list)
        for sv in self.SELECT_VARS:
            select_vars_by_table[sv.table].append(sv)

        self.valid = False
        df_list = []
        for table, select_vars in select_vars_by_table.items():
            c = conn.cursor()
            select_tokens = []
            for sv in select_vars:
                if sv.sql_select_str != sv.name:
                    select_tokens.append(sv.sql_select_str + ' AS ' + sv.name)
                else:
                    select_tokens.append(sv.sql_select_str)
            query = f"SELECT {', '.join(select_tokens)} FROM {table}"
            if table == 'metrics':
                query += " GROUP BY mcts_gen"
            c.execute(query)
            data = c.fetchall()
            if len(data) == 0:
                return
            columns = [sv.name for sv in select_vars]
            df = pd.DataFrame(data, columns=columns).set_index('mcts_gen')
            df_list.append(df)

        first_gen = max(df.index[0] for df in df_list)
        last_gen = min(df.index[-1] for df in df_list)

        full_df = df_list[0]
        for df in df_list[1:]:
            full_df = full_df.merge(df, how='outer', left_index=True, right_index=True)
        full_df = full_df.fillna(0)
        full_df = full_df[full_df.index <= last_gen]
        full_df = full_df[full_df.index >= first_gen]

        if len(full_df) == 0:
            return

        self.valid = True

        full_df['y_runtime'] *= 1e-9  # convert nanoseconds to seconds

        positions_evaluated = full_df['positions_evaluated']
        batches_evaluated = full_df['batches_evaluated'].replace(0, 1)

        cache_hits = full_df['cache_hits']
        cache_total = (cache_hits + full_df['cache_misses']).replace(0, 1)

        full_df['avg_batch_size'] = positions_evaluated / batches_evaluated
        full_df['cache_hit_rate'] = cache_hits / cache_total

        self.df = full_df

    def join(self, x_df: pd.DataFrame):
        x_df = x_df.reset_index()
        self.df = pd.merge(self.df, x_df, left_on='mcts_gen', right_on='mcts_gen',
                           how='inner').reset_index()


class SelfPlayPlotter:
    def __init__(self, data_list: List[SelfPlayData]):
        self.colors = bokeh_util.get_colors(len(data_list))
        self.data_list = data_list

        self.x_var_selector = XVarSelector([hd.df for hd in data_list], 'mcts_gen')
        self.sources: Dict[str, ColumnDataSource] = {}

        self.load()
        self.plots = [
            self.make_plot('avg_batch_size', 'Avg Batch Size', y_range_start=0),
            self.make_plot('cache_hit_rate', 'Cache Hit Rate'),
            # TODO: add more plots
            ]
        self.figure = self.make_figure()

    def load(self):
        for data in self.data_list:
            source = ColumnDataSource(data.df)
            self.x_var_selector.init_source(source)
            self.sources[data.tag] = source

    def make_figure(self):
        return column(*self.plots)

    def make_plot(self, y: str, descr: str, y_range_start=None):
        colors = self.colors

        plot = figure(
            title=descr, x_range=[0, 1],
            active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
        self.x_var_selector.init_plot(plot)

        for data, color in zip(self.data_list, colors):
            label = data.tag
            source = self.sources[label]
            plot.line('x', y, source=source, line_color=color, legend_label=label)

        hline = Span(location=0, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)
        if y_range_start is not None:
            plot.y_range.start = y_range_start

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'
        return plot
