from .x_var_logic import XVarSelector, make_x_df

from alphazero.dashboard.benchmark_plotting import BenchmarkPlotter
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


class EvaluationData:
    def __init__(self, run_params: RunParams, benchmark_tag: str):
        self.tag = run_params.tag
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        try:
            evaluator = Evaluator(organizer, benchmark_tag)
            benchmark_rating_data = evaluator._benchmark.read_ratings_from_db()
            eval_rating_data = evaluator.read_ratings_from_db()
        except Exception as e:
            print(f"Error loading evaluation for {self.tag}: {e}")
            self.valid = False
            return

        self.benchmark_gens = np.array([iagent.agent.gen for iagent in benchmark_rating_data.iagents])
        self.benchmark_tag = benchmark_rating_data.tag
        self.benchmark_ratings = benchmark_rating_data.ratings
        self.committee_gens = [evaluator._benchmark.indexed_agents[ix].agent.gen for ix in benchmark_rating_data.committee]

        self.evaluated_gens = np.array([ia.agent.gen for ia in eval_rating_data.evaluated_iagents])
        self.evaluated_ratings = eval_rating_data.ratings
        self.eval_tag = eval_rating_data.tag

        sorted_ix = np.argsort(self.benchmark_gens)
        self.df = pd.DataFrame({
            "mcts_gen": self.benchmark_gens[sorted_ix],
            "rating": self.benchmark_ratings[sorted_ix]
        })

        if self.benchmark_tag:
            benchmark_run_params = RunParams(game=run_params.game, tag=self.benchmark_tag)
            benchmark_organizer = DirectoryOrganizer(benchmark_run_params, base_dir_root='/workspace')
            benchmark_x_df = make_x_df(benchmark_organizer)
            self.df = self.df.merge(benchmark_x_df, left_on="mcts_gen", right_index=True, how="left")

        sorted_ix = np.argsort(self.evaluated_gens)
        self.eval_df = pd.DataFrame({
            "mcts_gen": self.evaluated_gens[sorted_ix],
            "rating": self.evaluated_ratings[sorted_ix]
        })
        x_df = make_x_df(organizer)
        self.eval_df = self.eval_df.merge(x_df, left_on="mcts_gen", right_index=True, how="left")

        self.valid = len(self.df) > 0


class EvaluationPlotter:
    def __init__(self, data: EvaluationData):
        self.data = data
        self.color = bokeh_util.get_colors(1)[0]
        self.plot = None
        self.layout = None
        self.make_plot()

    def valid(self):
        return self.data.valid

    def make_plot(self):
        df_eval = self.data.eval_df

        # Instantiate BenchmarkPlotter first
        benchmark_plotter = BenchmarkPlotter(self.data)
        if not benchmark_plotter.valid():
            self.plot = figure(title='No valid benchmark data')
            self.layout = column(self.plot)
            return

        # Use the benchmark's XVarSelector for axis sync
        self.x_selector = benchmark_plotter.x_selector

        # Setup eval source and sync x-axis
        self.eval_source = ColumnDataSource(df_eval)
        self.x_selector.init_source(self.eval_source)

        # Use benchmark plot as base
        plot = benchmark_plotter.plot

        # Update plot title for Evaluation
        plot.title.text = f"Evaluate '{self.data.tag}' against '{self.data.benchmark_tag}'"

        # Calculate x_range
        x_vals_benchmark = np.asarray(benchmark_plotter.source.data.get('x', []))
        x_vals_eval = np.asarray(self.eval_source.data.get('x', []))

        combined_x = np.concatenate([x_vals_benchmark, x_vals_eval])
        if len(combined_x) > 0:
            x_min, x_max = np.min(combined_x), np.max(combined_x)
            padding = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
            plot.x_range.start = x_min - padding
            plot.x_range.end = x_max + padding

        # Add eval line
        plot.line('x', 'rating', source=self.eval_source,
                line_width=1, color='red', legend_label='Evaluation')

        # Adjust y-range to cover both datasets
        y_min = min(self.data.df["rating"].min(), df_eval["rating"].min()) * 0.9
        y_max = max(self.data.df["rating"].max(), df_eval["rating"].max()) * 1.2
        plot.y_range.start = y_min
        plot.y_range.end = y_max

        # Shared radio group for both sources
        radio_group = self.x_selector.create_radio_group(
            [plot], [benchmark_plotter.source, self.eval_source]
        )
        self.plot = plot
        self.layout = column(plot, row(radio_group))

        # Sync markers on x-axis changes
        old_set_x_index = self.x_selector.set_x_index

        def new_set_x_index(x_index, plots, sources, force_refresh=False):
            old_set_x_index(x_index, plots, sources, force_refresh)
            benchmark_plotter.update_markers(self.x_selector.x_column)

            # Recalculate x_range after axis change
            x_vals_benchmark = np.asarray(benchmark_plotter.source.data.get('x', []))
            x_vals_eval = np.asarray(self.eval_source.data.get('x', []))

            combined_x = np.concatenate([x_vals_benchmark, x_vals_eval])
            if len(combined_x) > 0:
                x_min, x_max = np.min(combined_x), np.max(combined_x)
                padding = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
                plot.x_range.start = x_min - padding
                plot.x_range.end = x_max + padding

        self.x_selector.set_x_index = new_set_x_index
        benchmark_plotter.update_markers(self.x_selector.x_column)

def create_eval_figure(game: str, tag: str):
    run_params = RunParams(game=game, tag=tag)
    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

    if not os.path.exists(organizer.eval_db_dir):
        return figure(title='No evaluation data directory available')

    files = [f for f in os.listdir(organizer.eval_db_dir) if f.endswith('.db')]  # or appropriate extension
    if not files:
        return figure(title='No evaluation data files available')

    plots = []
    for f in files:
        benchmark_tag = os.path.splitext(f)[0]  # Strip file extension for tag

        data = EvaluationData(run_params, benchmark_tag)
        if not data.valid:
            print(f"Skipping invalid data from {f}")
            continue

        plotter = EvaluationPlotter(data)
        if not plotter.valid():
            print(f"Skipping invalid plot from {f}")
            continue

        plots.append(plotter.layout)

    if not plots:
        return figure(title='No valid evaluation plots found')

    return column(*plots)

