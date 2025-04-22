from .x_var_logic import XVarSelector, make_x_df
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span
from bokeh.layouts import column
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.run_params import RunParams
from alphazero.logic.benchmarker import Benchmarker
from util import bokeh_util
import numpy as np
import pandas as pd
import sqlite3
from typing import List, Dict
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, Select, CustomJS
from bokeh.layouts import column, row
from util import bokeh_util

class BenchmarkData:
    def __init__(self, run_params: RunParams):
        self.tag = run_params.tag
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        try:
            benchmarker = Benchmarker(organizer)
            rating_data = benchmarker.read_ratings_from_db()
        except Exception as e:
            print(f"Error loading benchmark for {self.tag}: {e}")
            self.valid = False
            return

        benchmark_gens = np.array([iagent.agent.gen for iagent in rating_data.iagents])
        benchmark_ratings = rating_data.ratings
        committee_gens = [benchmarker.indexed_agents[i].agent.gen for i in rating_data.committee]
        latest_gen = organizer.get_latest_model_generation()

        sorted_ix = np.argsort(benchmark_gens)
        gens_sorted = benchmark_gens[sorted_ix]
        ratings_sorted = benchmark_ratings[sorted_ix]

        self.df = pd.DataFrame({
            "mcts_gen": gens_sorted,
            "rating": ratings_sorted
        })

        x_df = make_x_df(organizer)
        self.df = self.df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")
        self.committee_gens = committee_gens
        self.benchmark_gens = benchmark_gens
        self.latest_gen = latest_gen
        self.valid = len(self.df) > 0


class SingleBenchmarkPlotter:
    def __init__(self, data: BenchmarkData):
        self.data = data
        self.source = ColumnDataSource(data.df)
        self.color = bokeh_util.get_colors(1)[0]
        self.plot = None
        self.layout = None
        self.make_plot()

    def valid(self):
        return self.data.valid

    def make_plot(self):
        df = self.data.df

        self.x_selector = XVarSelector([df])
        if not self.x_selector.valid:
            self.plot = figure(title='No valid x-axis data')
            self.layout = column(self.plot)
            return

        # Let XVarSelector set up the 'x' field
        self.source = ColumnDataSource(df)
        self.x_selector.init_source(self.source)

        x_df = np.asarray(self.source.data.get('x', []))
        x_min, x_max = np.min(x_df), np.max(x_df)
        padding = (x_max - x_min) * 0.05
        x_range = (x_min - padding, x_max + padding)

        # Create plot WITHOUT overriding x_range
        plot = figure(
            title=f'Benchmark Ratings - {self.data.tag}',
            y_axis_label='Rating',
            x_range=x_range,
            tools='pan,box_zoom,xwheel_zoom,reset,save'
        )

        if not self.x_selector.init_plot(plot):
            self.plot = figure(title='Failed to init x-axis')
            self.layout = column(self.plot)
            return

        # Now plot the line using whatever XVarSelector set as 'x'
        plot.line('x', 'rating', source=self.source,
                  line_width=2, line_color=self.color, legend_label='elo')

        # Annotations
        for gen in self.data.benchmark_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='orange',
                                line_dash='dashed', line_width=1))
        for gen in self.data.committee_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='green',
                                line_dash='dashed', line_width=2))

        # Dummy lines for legend
        plot.line(x=[0, 0], y=[0, 0], line_color='orange', line_dash='dashed', line_width=1,
                  legend_label="Evaluated Gens")
        plot.line(x=[0, 0], y=[0, 0], line_color='green', line_dash='dashed', line_width=2,
                  legend_label="Committee")

        plot.legend.location = 'top_left'
        plot.legend.click_policy = 'hide'
        plot.y_range.start = df["rating"].min() * 0.9
        plot.y_range.end = df["rating"].max() * 1.2

        # Let XVarSelector generate a proper radio group
        radio_group = self.x_selector.create_radio_group([plot], [self.source])
        self.plot = plot
        self.layout = column(plot, row(radio_group))


def create_benchmark_figure(game: str, tag: str):
    run_params = RunParams(game=game, tag=tag)
    data = BenchmarkData(run_params)
    if not data.valid:
        return figure(title='No benchmark data available')

    plotter = SingleBenchmarkPlotter(data)
    if not plotter.valid():
        return figure(title='No benchmark data available')

    return plotter.layout