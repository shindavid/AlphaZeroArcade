"""
Used by launch_dashboard.py to create an evaluation plot.
"""

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span
from bokeh.layouts import column
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.evaluator import Evaluator
from util import bokeh_util

import numpy as np
import pandas as pd


class EvaluationData:
    def __init__(self, run_params: RunParams):
        self.tag = run_params.tag
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        try:
            evaluator = Evaluator(organizer)
            benchmark_rating_data = evaluator._benchmark.read_ratings_from_db()
            eval_rating_data = evaluator.read_ratings_from_db()
        except Exception as e:
            print(f"Error loading evaluation for {self.tag}: {e}")
            self.valid = False
            return

        self.benchmark_gens = np.array([iagent.agent.gen for iagent in benchmark_rating_data.iagents])
        self.benchmark_ratings = benchmark_rating_data.ratings
        self.committee_gens = [evaluator._benchmark.indexed_agents[ix].agent.gen for ix in benchmark_rating_data.committee]

        self.evaluated_gens = np.array([ia.agent.gen for ia in eval_rating_data.evaluated_iagents])
        self.evaluated_ratings = eval_rating_data.ratings

        self.latest_gen = organizer.get_latest_model_generation()

        sorted_ix = np.argsort(self.benchmark_gens)
        self.df = pd.DataFrame({
            "gen": self.benchmark_gens[sorted_ix],
            "rating": self.benchmark_ratings[sorted_ix]
        })

        self.eval_df = pd.DataFrame({
            "gen": self.evaluated_gens,
            "rating": self.evaluated_ratings
        })

        self.valid = len(self.df) > 0


class EvaluationPlotter:
    def __init__(self, data: EvaluationData):
        self.data = data
        self.source = ColumnDataSource(data.df)
        self.eval_source = ColumnDataSource(data.eval_df)
        self.color = bokeh_util.get_colors(1)[0]
        self.plot = None
        self.layout = None
        self.make_plot()

    def valid(self):
        return self.data.valid

    def make_plot(self):
        plot = figure(
            title=f'Evaluation Plot - {self.data.tag}',
            x_axis_label='Generation',
            y_axis_label='Rating',
            tools='pan,box_zoom,xwheel_zoom,reset,save'
        )

        # Main ELO curve
        plot.line('gen', 'rating', source=self.source,
                  line_width=2, line_color=self.color, legend_label=self.data.tag)

        # Participated agents
        for gen in self.data.benchmark_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='white',
                                 line_dash='dashed', line_width=1))

        # Committee members
        for gen in self.data.committee_gens:
            plot.add_layout(Span(location=gen, dimension='height', line_color='orange',
                                 line_dash='dashed', line_width=2))

        # Latest model
        plot.add_layout(Span(location=self.data.latest_gen, dimension='height', line_color='green',
                             line_dash='dashed', line_width=2))

        # Evaluated test run (scatter in red)
        plot.scatter('gen', 'rating', source=self.eval_source,
                     size=8, color='red', legend_label='Test Run')

        # Dummy line for legend
        plot.line(x=[0, 0], y=[0, 0], line_color='white', line_dash='dashed', line_width=1,
                  legend_label="Evaluated Gens")
        plot.line(x=[0, 0], y=[0, 0], line_color='orange', line_dash='dashed', line_width=2,
                  legend_label="Committee")
        plot.line(x=[0, 0], y=[0, 0], line_color='green', line_dash='dashed', line_width=2,
                  legend_label="Latest Model")

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'

        plot.y_range.start = self.data.df["rating"].min() * 0.9
        plot.y_range.end = max(self.data.df["rating"].max(), self.eval_source.data["rating"].max()) * 1.2

        self.plot = plot
        self.layout = column(plot)


def create_eval_figure(game: str, tag: str):
    run_params = RunParams(game=game, tag=tag)
    data = EvaluationData(run_params)
    if not data.valid:
        return figure(title='No evaluation data available')

    plotter = EvaluationPlotter(data)
    if not plotter.valid():
        return figure(title='No evaluation data available')

    return plotter.layout
