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

from dataclasses import dataclass
import sqlite3
from typing import List


@dataclass
class XVar:
    label: str
    df_col: str
    sql_select_str: str


X_VARS = [
    XVar('Generation', 'mcts_gen', 'timestamps.gen'),
    XVar('Games', 'n_games', 'games'),
    XVar('Self-Play Runtime (sec)', 'runtime', 'end_timestamp - start_timestamp'),
    XVar('Num Evaluated Positions', 'n_evaluated_positions', 'positions_evaluated'),
    XVar('Num Evaluated Batches', 'n_batches_evaluated', 'batches_evaluated'),
    ]


def make_x_df(organizer: DirectoryOrganizer) -> pd.DataFrame:
    select_str = ', '.join([x_var.sql_select_str for x_var in X_VARS])
    query = f'SELECT {select_str} FROM self_play_metadata JOIN timestamps USING (gen)'
    conn = sqlite3.connect(organizer.self_play_db_filename)
    cursor = conn.cursor()
    values = cursor.execute(query).fetchall()
    conn.close()

    columns = [x_var.df_col for x_var in X_VARS]
    x_df = pd.DataFrame(values, columns=columns).set_index('mcts_gen')

    if len(x_df['runtime']) > 0:
        # Earlier versions stored runtimes in sec, not ns. This heuristic corrects the
        # earlier versions to ns.
        ts = max(x_df['runtime'])
        if ts > 1e9:
            x_df['runtime'] *= 1e-9  # ns -> sec

    for col in x_df:
        x_df[col] = x_df[col].cumsum()

    return x_df


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

    def init_plot(self, plot: figure):
        """
        Initializes the x-axis label and range of the given plot.
        """
        x_col = self.x_column
        plot.xaxis.axis_label = self.x_label
        plot.x_range.start = self._min_x_dict[x_col]
        plot.x_range.end = self._max_x_dict[x_col]

    def create_radio_group(self, plots: List[figure], sources: List[ColumnDataSource]) -> RadioGroup:
        labels = [x_var.label for x_var in X_VARS]
        radio_group = RadioGroup(labels=labels, active=self.x_index)

        def update_data(attr, old, new):
            x_index = radio_group.active
            self.set_x_index(x_index, plots, sources)

        radio_group.on_change('active', update_data)
        return radio_group

    def set_x_index(self, x_index: int, plots: List[figure], sources: List[ColumnDataSource]):
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

        if prev_x_index == x_index:
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

            if x_col == 'runtime':
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
