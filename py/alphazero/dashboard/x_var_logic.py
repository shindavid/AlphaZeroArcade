"""
For each dashboard plot, there are a number of choices for the x-axis variable.

This module contains shared logic for switching between these x-axis variables. The logic entails
details about tracking the current x-variable, and switching the x-range when the x-variable
changes.
"""
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

from bokeh.models import BasicTickFormatter, ColumnDataSource, CustomJSTickFormatter, RadioGroup
from bokeh.plotting import figure
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from typing import List


@dataclass
class XVar:
    db_filename_attr: str
    table: str
    label: str
    df_col: str
    sql_select_str: str


X_VARS = [
    XVar('self_play_db_filename', 'self_play_metadata', 'Self-Play Runtime',
         'runtime', 'runtime'),
    XVar('training_db_filename', 'training', 'Train Time',
         'train_time', 'training_end_ts - training_start_ts'),
    XVar('self_play_db_filename', 'self_play_metadata', 'Generation', 'mcts_gen', 'gen'),
    XVar('self_play_db_filename', 'self_play_metadata', 'Games', 'n_games', 'games'),
    XVar('self_play_db_filename', 'self_play_metadata', 'Num Evaluated Positions',
         'n_evaluated_positions', 'positions_evaluated'),
    XVar('self_play_db_filename', 'self_play_metadata', 'Num Evaluated Batches',
         'n_batches_evaluated', 'batches_evaluated'),
    ]


def make_x_df(organizer: DirectoryOrganizer) -> pd.DataFrame:
    x_var_dict = defaultdict(lambda: defaultdict(list))  # filename -> table -> XVar
    for x_var in X_VARS:
        x_var_dict[x_var.db_filename_attr][x_var.table].append(x_var)

    full_x_df = pd.DataFrame()
    for db_filename_attr, subdict in x_var_dict.items():
        db_filename = getattr(organizer, db_filename_attr)
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        for table, x_vars in subdict.items():
            select_str = ', '.join([x_var.sql_select_str for x_var in x_vars])
            query = f'SELECT {select_str} FROM {table}'
            values = cursor.execute(query).fetchall()
            columns = [x_var.df_col for x_var in x_vars]
            x_df = pd.DataFrame(values, columns=columns)
            full_x_df = pd.concat([full_x_df, x_df], axis=1)
        conn.close()

    full_x_df = full_x_df.set_index('mcts_gen')

    full_x_df['train_time'] *= 1e-9  # convert nanoseconds to seconds
    full_x_df['runtime'] *= 1e-9  # convert nanoseconds to seconds

    for col in full_x_df:
        full_x_df[col] = full_x_df[col].cumsum()

    return full_x_df


class XVarSelector:
    def __init__(self, df_list: List[pd.DataFrame]):
        """
        Each DataFrame in df_list is assumed to have a column for each str in X_VAR_COLUMNS.
        """
        self._x_index = 0
        self._min_x_dict = {}
        self._max_x_dict = {}

        for df in df_list:
            for x_var in X_VARS:
                x_col = x_var.df_col
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

    @property
    def x_label(self) -> str:
        return X_VARS[self._x_index].label

    @property
    def x_index(self) -> int:
        return self._x_index

    @property
    def x_column(self) -> str:
        return X_VARS[self._x_index].df_col

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
        x_col = self.x_column
        plot.xaxis.axis_label = self.x_label
        if x_col not in self._min_x_dict:
            return False
        plot.x_range.start = self._min_x_dict[x_col]
        plot.x_range.end = self._max_x_dict[x_col]
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
        x_col = self.x_column

        for source in sources:
            source.data['x'] = source.data[x_col]

        if not force_refresh and prev_x_index == x_index:
            return

        prev_x_min = self._min_x_dict[prev_x_col]
        prev_x_max = self._max_x_dict[prev_x_col]
        prev_x_width = prev_x_max - prev_x_min

        for plot in plots:
            plot.xaxis.axis_label = self.x_label
            start = plot.x_range.start
            end = plot.x_range.end
            if prev_x_width > 0:
                start_pct = (start - prev_x_min) / prev_x_width
                end_pct = (end - prev_x_min) / prev_x_width
                x_min = self._min_x_dict[x_col]
                x_max = self._max_x_dict[x_col]
                x_width = x_max - x_min
                plot.x_range.start = x_min + start_pct * x_width
                plot.x_range.end = x_min + end_pct * x_width

            if x_col in ('runtime', 'train_time'):
                plot.xaxis.formatter = CustomJSTickFormatter(code="""
                    var total_seconds = tick;
                    var days = Math.floor(total_seconds / 86400);
                    var hours = Math.floor((total_seconds % 86400) / 3600);
                    var minutes = Math.floor((total_seconds % 3600) / 60);
                    if (days > 0) {
                        return days + "d " + hours + "h " + minutes + "m";
                    } else {
                        return hours + "h " + minutes + "m";
                    }
                """)
            else:
                plot.xaxis.formatter = BasicTickFormatter()
