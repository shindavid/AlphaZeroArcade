from .x_var_logic import XVarSelector, make_x_df

from alphazero.dashboard.benchmark_plotting import BenchmarkPlotter
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span
from bokeh.layouts import gridplot, column, row
import numpy as np
import os
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


class Plotter:
    def __init__(self, data_list: List[EvaluationData]):
        self.benchmark_tag = data_list[0].benchmark_tag
        self.x_selector = XVarSelector([data.df for data in data_list])
        self.sources: Dict[str, ColumnDataSource] = {}
        self.min_y = 0
        self.max_y = 0
        self.load(data_list)
        self.plotted_labels = set()
        self.figure = self.make_figure()

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

    def make_figure(self):
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
    if not data_list:
        return figure(title='No data available')
    plotter = Plotter(data_list)
    return plotter.figure

