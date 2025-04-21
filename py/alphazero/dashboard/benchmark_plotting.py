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
from bokeh.layouts import column
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
        print(f"Latest model generation: {latest_gen}")

        sorted_ix = np.argsort(benchmark_gens)
        gens_sorted = benchmark_gens[sorted_ix]
        ratings_sorted = benchmark_ratings[sorted_ix]

        self.df = pd.DataFrame({
            "gen": gens_sorted,
            "rating": ratings_sorted
        })

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
        plot = figure(
            title=f'Benchmark Ratings - {self.data.tag}',
            x_axis_label='Generation',
            y_axis_label='Rating',
            tools='pan,box_zoom,xwheel_zoom,reset,save'
        )

        # Rating line
        plot.line('gen', 'rating', source=self.source,
                  line_width=2, line_color=self.color, legend_label='elo')

        # Evaluated gens (white)
        for gen in self.data.benchmark_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='white',
                                 line_dash='dashed', line_width=1))

        # Committee members (orange)
        for gen in self.data.committee_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='orange',
                                 line_dash='dashed', line_width=2))

        # Latest model (green)
        plot.add_layout(Span(location=self.data.latest_gen, dimension='height', line_color='green',
                             line_dash='dashed', line_width=3))

        plot.line(x=[0, 0], y=[0, 0], line_color='white', line_dash='dashed', line_width=1,
                  legend_label="Evaluated Gens")
        plot.line(x=[0, 0], y=[0, 0], line_color='orange', line_dash='dashed', line_width=2,
                  legend_label="Committee")
        plot.line(x=[0, 0], y=[0, 0], line_color='green', line_dash='dashed', line_width=3,
                  legend_label="Latest Model")

        plot.legend.location = 'top_left'
        plot.legend.click_policy = 'hide'

        plot.y_range.start = self.data.df["rating"].min() * 0.9
        plot.y_range.end = self.data.df["rating"].max() * 1.2

        self.plot = plot
        self.layout = column(plot)


def create_benchmark_figure(game: str, tag: str):
    run_params = RunParams(game=game, tag=tag)
    data = BenchmarkData(run_params)
    if not data.valid:
        return figure(title='No benchmark data available')

    plotter = SingleBenchmarkPlotter(data)
    if not plotter.valid():
        return figure(title='No benchmark data available')

    return plotter.layout