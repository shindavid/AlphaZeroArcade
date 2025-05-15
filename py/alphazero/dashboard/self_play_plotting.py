"""
Used by launch_dashboard.py to create a self-play plot.
"""
from .x_var_logic import XVarSelector, make_x_df
from alphazero.logic import constants
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Span, Select
from bokeh.plotting import figure
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from typing import Callable, Dict, List, Optional


Tag = str


def create_self_play_figure(game: str, tags: List[Tag]):
    data_dict: Dict[Tag, SelfPlayData] = {}
    for tag in tags:
        run_params = RunParams(game, tag)
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        conn = sqlite3.connect(organizer.self_play_db_filename)
        data = SelfPlayData(conn, tag)
        conn.close()
        if not data.valid:
            continue

        x_df = make_x_df(organizer)
        data.join(x_df)
        data_dict[tag] = data

    if not data_dict:
        return figure(title='No data available')

    return SelfPlayPlotter(data_dict).figure


@dataclass
class SelectVar:
    table: str
    df_col: str
    sql_select_str: str
    group_by: Optional['str'] = None
    index: bool = False


SELECT_VARS = [
    SelectVar('metrics', 'mcts_gen', 'gen', index=True),
]

for col in constants.PERF_STATS_COLUMNS:
    func = 'MAX' if col == 'batch_datas_allocated' else 'SUM'
    SELECT_VARS.append(
        SelectVar('metrics', col, f'{func}({col})', group_by='gen'))


def div(a, b):
    return a / b.replace(0, 1)


@dataclass
class YVar:
    descr: str
    column: str
    range_start: Optional[float] = None
    range_end: Optional[float] = None
    func: Optional[Callable[[pd.DataFrame], pd.Series]] = None


BATCH_SIZE_PLOT = [
    YVar('Avg Batch Size', 'avg_batch_size', range_start=0, func=lambda df: div(
        df['positions_evaluated'], df['batches_evaluated'])),
]

CACHE_HIT_RATE_PLOT = [
    YVar('Cache Hit Rate', 'cache_hit_rate', range_start=0, range_end=1, func=lambda df: div(
        df['cache_hits'], df['cache_hits'] + df['cache_misses'])),
]

BATCHES_ALLOCATED_PLOT = [
    YVar('Batches Allocated', 'batch_datas_allocated_cs', range_start=0,
         func=lambda df: df['batch_datas_allocated'].cumsum()),
]

STACKED_TOTAL_RUNTIME_PLOT = [
    YVar('pause', 'pause_time_s', func=lambda df: df['pause_time_ns'] * 1e-9),
    YVar('reload', 'model_load_time_s', func=lambda df: df['model_load_time_ns'] * 1e-9),
    YVar('work', 'work_time_s', func=lambda df:
        (df['total_time_ns'] - df['pause_time_ns'] - df['model_load_time_ns']) * 1e-9),
]

STACKED_NN_EVAL_PER_BATCH_PLOT = [
    YVar('cpu->gpu', 'cpu2gpu_copy_time_ms',
         func=lambda df: df['cpu2gpu_copy_time_ns'] / (1e6 * df['batches_evaluated'])),
    YVar('gpu->cpu', 'gpu2cpu_copy_time_ms',
         func=lambda df: df['gpu2cpu_copy_time_ns'] / (1e6 * df['batches_evaluated'])),
    YVar('eval', 'model_eval_time_ms',
         func=lambda df: df['model_eval_time_ns'] / (1e6 * df['batches_evaluated'])),
    YVar('wait', 'wait_for_search_threads_time_ms',
        func=lambda df: df['wait_for_search_threads_time_ns'] / (1e6 * df['batches_evaluated'])),
]

STACKED_SEARCH_THREAD_PLOT = [
    YVar('wait-slot', 'wait_slot_s', func=lambda df: df['wait_for_game_slot_time_ns'] * 1e-9),
    YVar('cache-mutex', 'cache_mutex_s', func=lambda df: df['cache_mutex_acquire_time_ns'] * 1e-9),
    YVar('cache-insert', 'cache_insert_s', func=lambda df: df['cache_insert_time_ns'] * 1e-9),
    YVar('batch-prepare', 'batch_prepare_s', func=lambda df: df['batch_prepare_time_ns'] * 1e-9),
    YVar('batch-write', 'batch_write_s', func=lambda df: df['batch_write_time_ns'] * 1e-9),
    YVar('wait-nn', 'wait_nn_s', func=lambda df: df['wait_for_nn_eval_time_ns'] * 1e-9),
    YVar('mcts', 'mcts_s', func=lambda df: df['mcts_time_ns'] * 1e-9),
]


class SelfPlayData:
    def __init__(self, conn: sqlite3.Connection, tag: str):
        self.tag = tag
        select_vars_by_table = defaultdict(list)
        for sv in SELECT_VARS:
            select_vars_by_table[sv.table].append(sv)

        self.valid = False
        df_list = []
        for table, select_vars in select_vars_by_table.items():
            c = conn.cursor()
            select_tokens = []
            for sv in select_vars:
                if sv.sql_select_str != sv.df_col:
                    select_tokens.append(sv.sql_select_str + ' AS ' + sv.df_col)
                else:
                    select_tokens.append(sv.sql_select_str)
            query = f"SELECT {', '.join(select_tokens)} FROM {table} WHERE gen > 0"
            group_by = set([sv.group_by for sv in select_vars if sv.group_by is not None])
            if group_by:
                assert len(group_by) == 1, select_vars
                query += f" GROUP BY {group_by.pop()}"

            c.execute(query)
            data = c.fetchall()
            if len(data) == 0:
                return

            index_vars = [sv for sv in select_vars if sv.index]
            assert len(index_vars) == 1, (table, select_vars)
            index = index_vars[0].df_col

            columns = [sv.df_col for sv in select_vars]
            df = pd.DataFrame(data, columns=columns).set_index(index)
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

        for plot in [STACKED_TOTAL_RUNTIME_PLOT, STACKED_NN_EVAL_PER_BATCH_PLOT,
                     STACKED_SEARCH_THREAD_PLOT, BATCH_SIZE_PLOT, CACHE_HIT_RATE_PLOT,
                     BATCHES_ALLOCATED_PLOT]:
            for y_var in plot:
                assert y_var.column not in full_df.columns, y_var
                full_df[y_var.column] = y_var.func(full_df)

        # chop off the last row because partial generations are misleading
        full_df = full_df.iloc[:-1]

        self.df = full_df

    def join(self, x_df: pd.DataFrame):
        x_df = x_df.reset_index()
        self.df = pd.merge(self.df, x_df, left_on='mcts_gen', right_on='mcts_gen',
                           how='inner').reset_index()


class SelfPlayPlotter:
    def __init__(self, data_dict: Dict[Tag, SelfPlayData]):
        self.data_dict = data_dict
        self.x_var_selector = XVarSelector([hd.df for hd in data_dict.values()], 'mcts_gen')
        self.sources: Dict[str, ColumnDataSource] = {}

        self.load()
        self.figure = self.make_figure()

    def load(self):
        for data in self.data_dict.values():
            source = ColumnDataSource(data.df)
            self.x_var_selector.init_source(source)
            self.sources[data.tag] = source

    def make_figure(self):
        self.plots = []
        self.plots.extend(self.make_stacked_plots(STACKED_TOTAL_RUNTIME_PLOT, 'Total Runtime',
                                                  'Seconds'))
        self.plots.extend(self.make_stacked_plots(STACKED_NN_EVAL_PER_BATCH_PLOT,
                                                  'NN Eval Profiling - Per Batch', 'Milliseconds'))
        self.plots.extend(self.make_stacked_plots(STACKED_SEARCH_THREAD_PLOT,
                                                  'Search Thread Profiling', 'Seconds'))
        self.plots.extend([self.make_single_plot(BATCH_SIZE_PLOT),
            self.make_single_plot(CACHE_HIT_RATE_PLOT),
            self.make_single_plot(BATCHES_ALLOCATED_PLOT),
        ])


        return column(*self.plots)

    def make_single_plot(self, y_var_list: List[YVar]):
        assert len(y_var_list) == 1
        y_var = y_var_list[0]
        y = y_var.column
        descr = y_var.descr
        colors = bokeh_util.get_colors(len(self.data_dict))

        plot = figure(
            title=descr, x_range=[0, 1],
            active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
        self.x_var_selector.init_plot(plot)

        for data, color in zip(self.data_dict.values(), colors):
            label = data.tag
            source = self.sources[label]
            plot.line('x', y, source=source, line_color=color, legend_label=label)

        hline = Span(location=0, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)

        if y_var.range_start is not None:
            plot.y_range.start = y_var.range_start
        if y_var.range_end is not None:
            plot.y_range.end = y_var.range_end

        if y_var.descr is not None:
            plot.yaxis.axis_label = y_var.descr

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'
        return plot

    def make_stacked_plots(self, y_var_list: List[YVar], descr: str, y_axis_label: str = None):
        for data in self.data_dict.values():
            label = data.tag
            title = f'{descr} ({label})'
            plot = figure(
                title=title, x_range=[0, 1],
                active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
            self.x_var_selector.init_plot(plot)

            colors = bokeh_util.get_colors(len(y_var_list))
            source = self.sources[label]

            y_cols = [y_var.column for y_var in y_var_list]
            legend_labels = [y_var.descr for y_var in y_var_list]
            bokeh_util.add_stacked_area(plot, source, y_cols, 'x', colors, legend_labels)

            hline = Span(location=0, dimension='width', line_color='gray',
                        line_dash='dashed', line_width=1)
            plot.add_layout(hline)

            if y_axis_label is not None:
                plot.yaxis.axis_label = y_axis_label

            plot.legend.items = list(reversed(plot.legend.items))
            plot.legend.location = 'bottom_left'
            yield plot
