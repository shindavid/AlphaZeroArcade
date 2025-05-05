"""
For each dashboard plot, there are a number of choices for the x-axis variable.

This module contains shared logic for switching between these x-axis variables. The logic entails
details about tracking the current x-variable, and switching the x-range when the x-variable
changes.
"""
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.bokeh_util import make_time_tick_formatter

from bokeh.models import BasicTickFormatter, ColumnDataSource, RadioGroup
from bokeh.plotting import figure
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from typing import Callable, List, Optional


@dataclass
class SelectVar:
    db_filename_attr: str
    table: str
    df_col: str
    sql_select_str: str
    index: bool = False


SELECT_VARS = [
    SelectVar('self_play_db_filename', 'metrics', 'runtime',
              '1e-9 * (total_time_ns - pause_time_ns)'),
    SelectVar('self_play_db_filename', 'metrics', 'mcts_gen', 'gen', index=True),
    SelectVar('self_play_db_filename', 'self_play_data', 'mcts_gen', 'gen', index=True),
    SelectVar('self_play_db_filename', 'self_play_data', 'n_games', 'games'),
    SelectVar('self_play_db_filename', 'self_play_data', 'n_evaluated_positions',
              'positions_evaluated'),
    SelectVar('self_play_db_filename', 'self_play_data', 'n_batches_evaluated',
              'batches_evaluated'),
    SelectVar('training_db_filename', 'training', 'train_time',
              '1e-9 * (training_end_ts - training_start_ts)'),
    SelectVar('training_db_filename', 'training', 'mcts_gen', 'gen', index=True),
]


@dataclass
class XVar:
    label: str
    column: str
    apply_cumsum: bool = True
    is_time_var: bool = False
    func: Optional[Callable[[pd.DataFrame], pd.Series]] = None


X_VARS = [
    XVar('Total GPU time', 'total_time', is_time_var=True,
         func=lambda df: df['runtime'] + df['train_time']),
    XVar('Self-Play Runtime', 'runtime', is_time_var=True),
    XVar('Train Time', 'train_time', is_time_var=True),
    XVar('Generation', 'mcts_gen', apply_cumsum=False),
    XVar('Games', 'n_games'),
    XVar('Num Evaluated Positions', 'n_evaluated_positions'),
]


def make_x_df(organizer: DirectoryOrganizer) -> pd.DataFrame:
    select_var_dict = defaultdict(lambda: defaultdict(list))  # filename -> table -> XVar
    for select_var in SELECT_VARS:
        select_var_dict[select_var.db_filename_attr][select_var.table].append(select_var)

    x_df_list = []
    for db_filename_attr, subdict in select_var_dict.items():
        db_filename = getattr(organizer, db_filename_attr)
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        for table, select_vars in subdict.items():
            select_strs = [sv.sql_select_str for sv in select_vars]
            columns = [sv.df_col for sv in select_vars]

            select_str = ', '.join(select_strs)
            query = f'SELECT {select_str} FROM {table}'
            values = cursor.execute(query).fetchall()

            index_vars = [sv for sv in select_vars if sv.index]
            assert len(index_vars) == 1, (db_filename_attr, table, select_vars)
            index = index_vars[0].df_col
            x_df = pd.DataFrame(values, columns=columns).set_index(index)
            x_df_list.append(x_df)
        conn.close()

    full_x_df = x_df_list[0]
    for x_df in x_df_list[1:]:
        full_x_df = full_x_df.merge(x_df, how='outer', left_index=True, right_index=True)
    full_x_df = full_x_df.fillna(0).infer_objects(copy=False)

    for x_var in X_VARS:
        if x_var.func is not None:
            full_x_df[x_var.column] = x_var.func(full_x_df)
        if x_var.apply_cumsum:
            full_x_df[x_var.column] = full_x_df[x_var.column].cumsum()

    return full_x_df


class XVarSelector:
    def __init__(self, df_list: List[pd.DataFrame], initial_df_col: Optional[str] = None):
        """
        Each DataFrame in df_list is assumed to have a column for each str in X_VAR_COLUMNS.
        """
        if initial_df_col is None:
            initial_df_col = X_VARS[0].column
        self._x_index = None
        self._min_x_dict = {}
        self._max_x_dict = {}

        for df in df_list:
            for x_index, x_var in enumerate(X_VARS):
                x_col = x_var.column
                if x_col == initial_df_col:
                    assert self._x_index in (None, x_index)
                    self._x_index = x_index
                assert x_col in df.columns, f"Column '{x_col}' not found in DataFrame:\n{df}"
                x = df[x_col]
                if len(x) == 0:
                    continue
                mx = min(x)
                self._min_x_dict[x_col] = mx if x_col not in self._min_x_dict else \
                    min(self._min_x_dict[x_col], mx)
                mx = max(x)
                self._max_x_dict[x_col] = mx if x_col not in self._max_x_dict else \
                    max(self._max_x_dict[x_col], mx)

    def valid(self) -> bool:
        return self._x_index is not None

    @property
    def x_label(self) -> str:
        return X_VARS[self._x_index].label

    @property
    def x_index(self) -> int:
        return self._x_index

    @property
    def x_column(self) -> str:
        return X_VARS[self._x_index].column

    def init_source(self, source: ColumnDataSource):
        """
        Initializes source.data['x'] to the x-column of the source's DataFrame.
        """
        source.data['x'] = source.data[self.x_column]

    def init_plot(self, plot: figure) -> bool:
        """
        Initializes the x-axis label and range of the given plot.

        Returns True if the initialization succeeds.
        """
        if self.x_index is None:
            return False
        x_col = self.x_column
        plot.xaxis.axis_label = self.x_label
        if x_col not in self._min_x_dict:
            return False
        padding = (self._max_x_dict[x_col] - self._min_x_dict[x_col]) * 0.05
        plot.x_range.start = self._min_x_dict[x_col] - padding
        plot.x_range.end = self._max_x_dict[x_col] + padding
        return True

    def create_radio_group(self, plots: List[figure], sources: List[ColumnDataSource]) -> RadioGroup:
        labels = [x_var.label for x_var in X_VARS]
        radio_group = RadioGroup(labels=labels, active=self.x_index)

        def update_data(attr, old, new):
            x_index = radio_group.active
            self.set_x_index(x_index, plots, sources)

        self.set_x_index(self._x_index, plots, sources, force_refresh=True)
        radio_group.on_change('active', update_data)
        return radio_group

    def set_x_index(self, x_index: int, plots: List[figure], sources: List[ColumnDataSource],
                    force_refresh=False):
        """
        Performs the following:

        1. Sets the x-variable to the given index
        2. Updates the axis-label and x-range of each plot in plots
        3. Updates source.data['x'] for each source in sources
        """
        prev_x_index = self._x_index
        prev_x_col = self.x_column

        self._x_index = x_index
        x_var = X_VARS[x_index]
        x_col = self.x_column

        for source in sources:
            source.data['x'] = source.data[x_col]

        if not force_refresh and prev_x_index == x_index:
            return

        if prev_x_col not in self._min_x_dict:
            return

        prev_x_min = self._min_x_dict[prev_x_col]
        prev_x_max = self._max_x_dict[prev_x_col]
        prev_x_width = prev_x_max - prev_x_min

        for plot in plots:
            plot.xaxis.axis_label = self.x_label
            start = plot.x_range.start
            end = plot.x_range.end

            if start is None or end is None or prev_x_width <= 0:
                # fallback to full range
                x_min = self._min_x_dict[x_col]
                x_max = self._max_x_dict[x_col]
                plot.x_range.start = x_min
                plot.x_range.end = x_max
            else:
                start_pct = (start - prev_x_min) / prev_x_width
                end_pct = (end - prev_x_min) / prev_x_width

                x_min = self._min_x_dict[x_col]
                x_max = self._max_x_dict[x_col]
                x_width = x_max - x_min

                plot.x_range.start = x_min + start_pct * x_width
                plot.x_range.end = x_min + end_pct * x_width

            if x_var.is_time_var:
                plot.xaxis.formatter = make_time_tick_formatter(resolution='m')
            else:
                plot.xaxis.formatter = BasicTickFormatter()
