"""
This module handles the plotting of evaluation data for the "Evaluation" tab in the dashboard.

When runs are evaluated against a benchmark, only their evaluation data is plotted. Benchmark data
from the reference run (i.e., /output/{game}/{tag}/databases/evaluation/{benchmark_tag}.db) will be
included only if evaluation data for that benchmark exists.

This design choice helps avoid misleading plots when the benchmark run has not been evaluated on a
comparable cadence with other runs. For instance, if the benchmark run is evaluated more densely across
generations, its Elo curve may appear to achieve higher ratings sooner—not because it performed better,
but because its performance was measured more frequently. This can falsely suggest superior performance
compared to other runs.

If no other runs have been evaluated against the benchmark, only the benchmark's Elo data will be plotted.
"""

from .x_var_logic import XVarSelector, make_x_df

from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
import numpy as np
import pandas as pd
from typing import Dict, List


class EvaluationData:
    def __init__(self, run_params: RunParams, benchmark_tag: str):
        self.tag = run_params.tag
        self.benchmark_tag = benchmark_tag

        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
        self.df = self.make_df(organizer, benchmark_tag)

        x_df = make_x_df(organizer)
        self.df = self.df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")
        self.valid = len(self.df) > 0

    def make_df(self, organizer: DirectoryOrganizer, benchmark_tag: str):
        try:
            evaluator = Evaluator(organizer, benchmark_tag)
            eval_rating_data = evaluator.read_ratings_from_db()
        except Exception as e:
            print(f"Error loading evaluation for {self.tag}: {e}")
            self.valid = False
            return

        evaluated_gens = np.array([ia.agent.gen for ia in eval_rating_data.evaluated_iagents])
        evaluated_ratings = eval_rating_data.ratings
        sorted_ix = np.argsort(evaluated_gens)

        df = pd.DataFrame({
            "mcts_gen": evaluated_gens[sorted_ix],
            "rating": evaluated_ratings[sorted_ix]
        })

        return df


def get_eval_data_list(game: str, benchmark_tag: str, tags: List[str]) -> List[EvaluationData]:
    data_list = []
    for tag in tags:
        run_params = RunParams(game=game, tag=tag)
        data = EvaluationData(run_params, benchmark_tag)
        if data.valid:
            data_list.append(data)
    return data_list


class BenchmarkData:
    def __init__(self, run_params: RunParams):
        self.tag = run_params.tag

        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
        self.df = self.make_df(organizer)

        x_df = make_x_df(organizer)
        self.df = self.df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")
        self.valid = len(self.df) > 0

    def make_df(self, organizer: DirectoryOrganizer):
        try:
            benchmarker = Benchmarker(organizer)
            benchmark_rating_data = benchmarker.read_ratings_from_db()
        except Exception as e:
            print(f"Error loading benchmark for {self.tag}: {e}")
            self.valid = False
            return

        benchmark_gens = np.array([iagent.agent.gen for iagent in benchmark_rating_data.iagents])
        benchmark_ratings = benchmark_rating_data.ratings

        sorted_ix = np.argsort(benchmark_gens)
        gens_sorted = benchmark_gens[sorted_ix]
        ratings_sorted = benchmark_ratings[sorted_ix]

        df = pd.DataFrame({
            "mcts_gen": gens_sorted,
            "rating": ratings_sorted
        })

        return df


class Plotter:
    def __init__(self, data_list: List[EvaluationData], benchmark_data: BenchmarkData = None):
        if not data_list:
            data_list = [benchmark_data]
            self.benchmark_tag = benchmark_data.tag
        else:
            self.benchmark_tag = data_list[0].benchmark_tag

        self.x_selector = XVarSelector([data.df for data in data_list])
        self.sources: Dict[str, ColumnDataSource] = {}
        self.min_y = 0
        self.max_y = 0
        self.load(data_list)
        self.plotted_labels = set()
        self.figure = self.make_eval_figure()

    def load(self, data_list: List[EvaluationData]):
        self.data_list = data_list
        for data in data_list:
            df = data.df
            source = ColumnDataSource(df)
            source.data['y'] = df['rating']
            self.x_selector.init_source(source)
            self.sources[data.tag] = source
            self.max_y = max(self.max_y, max(df['rating']))
            self.min_y = min(self.min_y, min(df['rating']))

    def add_lines(self, plot):
        data_list = self.data_list
        n = len(data_list)
        colors = bokeh_util.get_colors(n+1)
        for data, color in zip(data_list, colors):
            label = data.tag
            if label in self.plotted_labels:
                continue
            self.plotted_labels.add(label)
            source = self.sources[label]
            plot.line('x', 'y', source=source, line_width=1, color=color, legend_label=label)

    def make_eval_figure(self):
        padding = (self.max_y - self.min_y) * 0.05
        y_range = [self.min_y - padding, self.max_y + padding]
        title = f'Evaluation Ratings on benchmark: {self.benchmark_tag}'
        plot = figure(title=title, x_range=[0, 1], y_range=y_range, y_axis_label='Rating',
                      active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')

        if not self.x_selector.init_plot(plot):
            return None

        self.add_lines(plot)

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'

        radio_group = self.x_selector.create_radio_group([plot], list(self.sources.values()))

        return column(plot, radio_group)


def create_eval_figure(game: str, benchmark_tag: str, tags: List[str]):
    data_list = get_eval_data_list(game, benchmark_tag, tags)
    benchmark_data = BenchmarkData(RunParams(game=game, tag=benchmark_tag))
    plotter = Plotter(data_list, benchmark_data)
    return plotter.figure
