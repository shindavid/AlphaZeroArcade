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

from alphazero.logic.agent_types import AgentRole
from alphazero.logic.self_evaluator import SelfEvaluator
from alphazero.logic.rating_db import DBAgentRating, RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Benchmark, Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Span
from bokeh.layouts import column, row
import numpy as np
import pandas as pd

import os
import sqlite3
from typing import Dict, List


class RatingData:
    @property
    def label(self):
        if not self.tag:
            raise ValueError("tag is not set")

        if self.rating_tag:
            return f'{self.tag}: {self.rating_tag}'
        return self.tag

class EvaluationData(RatingData):
    def __init__(self, run_params: RunParams, benchmark_tag: str, rating_tag: str):
        self.tag = run_params.tag
        self.benchmark_tag = benchmark_tag
        self.rating_tag = rating_tag

        organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
        self.benchmark_elos = {}
        self.df = self.make_df(organizer, benchmark_tag, rating_tag)

        if self.df is not None:
            x_df = make_x_df(organizer)
            self.df = self.df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")
            self.valid = len(self.df) > 0

    def make_df(self, organizer: DirectoryOrganizer, benchmark_tag: str, rating_tag: str):
        try:
            db = RatingDB(organizer.eval_db_filename(benchmark_tag))
            eval_ratings: List[DBAgentRating] = db.load_ratings(AgentRole.TEST)
            benchmark_ratings: List[DBAgentRating] = db.load_ratings(AgentRole.BENCHMARK)

            if len(benchmark_ratings) > 0:
                for data in benchmark_ratings:
                    self.benchmark_elos[data.level] = data.rating

            gen_ratings = [(data.level, data.rating) for data in eval_ratings if data.rating_tag == rating_tag]
            evaluated_gens, evaluated_ratings = zip(*gen_ratings)
            evaluated_gens = np.array(evaluated_gens, dtype=int)
            evaluated_ratings = np.array(evaluated_ratings, dtype=float)
            sorted_ix = np.argsort(evaluated_gens)

            df = pd.DataFrame({
                "mcts_gen": evaluated_gens[sorted_ix],
                "rating": evaluated_ratings[sorted_ix]
            })

        except Exception as e:
            self.valid = False
            return None

        return df


def make_eval_data_list(run_params: RunParams, benchmark_tag: str) -> List[EvaluationData]:
    organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
    db_filename = organizer.eval_db_filename(benchmark_tag)
    if not os.path.exists(db_filename):
        return []

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    res = cursor.execute('SELECT DISTINCT rating_tag FROM evaluator_ratings')
    rating_tags = [r[0] for r in res.fetchall()]
    conn.close()

    data_list = [EvaluationData(run_params, benchmark_tag, t) for t in rating_tags]
    return [d for d in data_list if d.valid]


def get_eval_data_list(game: str, benchmark_tag: str, tags: List[str]) -> List[EvaluationData]:
    data_list = []
    for tag in tags:
        run_params = RunParams(game=game, tag=tag)
        data_list.extend(make_eval_data_list(run_params, benchmark_tag))

    return data_list


class BenchmarkData(RatingData):
    def __init__(self, organizer: DirectoryOrganizer):
        self.benchmark_elos = {}
        self.df = self.make_df(organizer)
        self.tag = organizer.tag

        x_df = make_x_df(organizer)
        self.df = self.df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")
        self.valid = len(self.df) > 0

    def make_df(self, organizer: DirectoryOrganizer):
        try:
            self_evaluator = SelfEvaluator(organizer)
            benchmark_rating_data = self_evaluator.read_ratings_from_db()
            ratings = benchmark_rating_data.ratings
            iagents = benchmark_rating_data.iagents
            for ia, rating in zip(iagents, ratings):
                self.benchmark_elos[ia.agent.level] = rating

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
    def __init__(self, data_list: List[EvaluationData], benchmark_data: BenchmarkData):
        if not data_list:
            if benchmark_data is None:
                self.figure = None
                return
            data_list = [benchmark_data]
            self.benchmark_tag = benchmark_data.tag
        else:
            self.benchmark_tag = data_list[0].benchmark_tag

        self.benchmark_data = benchmark_data
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
            self.sources[data.label] = source
            self.max_y = max(self.max_y, max(df['rating']))
            self.min_y = min(self.min_y, min(df['rating']))

        self.benchmark_elos = self.data_list[0].benchmark_elos
        max_benchmark_elo = max(self.benchmark_elos.values(), default=0)
        self.max_y = max(self.max_y, max_benchmark_elo)

    def add_lines(self, plot):
        data_list = self.data_list
        n = len(data_list)
        colors = bokeh_util.get_colors(n+1)
        for data, color in zip(data_list, colors):
            label = data.label
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

        if self.benchmark_tag == 'reference.player':
            level_keys = sorted(self.benchmark_elos.keys(), key=lambda x: int(x))
            initial_level = level_keys[-1] if self.benchmark_elos else None

            hline = Span(
                location=self.benchmark_elos[initial_level],
                dimension='width',
                line_color='green',
                line_width=2,
                line_dash='dashed'
                )

            plot.add_layout(hline)

            slider = Slider(
                start=int(level_keys[0]),
                end=int(level_keys[-1]),
                value=int(initial_level),
                step=1,
                title="level"
            )

            def update_hline(attr, old, new):
                new_elo = self.benchmark_elos.get(new, None)
                hline.location = new_elo

            slider.on_change("value", update_hline)
            return column(plot, row(radio_group, slider))

        else:
            benchmark_source = ColumnDataSource(self.benchmark_data.df)
            plot.scatter(
                x='x',
                y='rating',
                source=benchmark_source,
                size=8,
                color='grey',
                legend_label=self.benchmark_tag,
                marker='circle'
            )
            radio_group = self.x_selector.create_radio_group(
                [plot], list(self.sources.values()) + [benchmark_source])

        return column(plot, radio_group)


def create_eval_figure(game: str, benchmark_tag: str, tags: List[str]):
    data_list = get_eval_data_list(game, benchmark_tag, tags)
    if RunParams.is_valid_tag(benchmark_tag):
        organizer = DirectoryOrganizer(RunParams(game, benchmark_tag), base_dir_root=Workspace)
        if os.path.exists(organizer.benchmark_db_filename):
            benchmark_organizer = organizer
        else:
            run_params = RunParams(game, benchmark_tag)
            benchmark_organizer = DirectoryOrganizer(run_params, base_dir_root=Benchmark)
        benchmark_data = BenchmarkData(benchmark_organizer)
    else:
        benchmark_data = None
    plotter = Plotter(data_list, benchmark_data)
    return plotter.figure
