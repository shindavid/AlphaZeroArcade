from .x_var_logic import XVarSelector, make_x_df
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span
from bokeh.layouts import column
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.evaluator import Evaluator
from util import bokeh_util
from bokeh.layouts import column, row

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

        self.eval_df = pd.DataFrame({
            "mcts_gen": self.evaluated_gens,
            "rating": self.evaluated_ratings
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
        df_benchmark = self.data.df
        df_eval = self.data.eval_df

        # Single XVarSelector for both sources
        self.x_selector = XVarSelector([df_benchmark, df_eval])
        if not self.x_selector.valid:
            self.plot = figure(title='No valid x-axis data')
            self.layout = column(self.plot)
            return

        self.source = ColumnDataSource(df_benchmark)
        self.eval_source = ColumnDataSource(df_eval)

        # Apply x-axis logic to both sources
        self.x_selector.init_source(self.source)
        self.x_selector.init_source(self.eval_source)

        # Get initial 'x' data (after x_selector added it)
        x_vals_benchmark = np.asarray(self.source.data.get('x', []))
        x_vals_eval = np.asarray(self.eval_source.data.get('x', []))

        # Combine and compute min/max
        all_x = np.concatenate([x_vals_benchmark, x_vals_eval])
        x_min, x_max = np.min(all_x), np.max(all_x)
        padding = (x_max - x_min) * 0.05
        x_range = (x_min - padding, x_max + padding)

        plot = figure(
            title=f'Evaluation Plot - {self.data.tag}',
            y_axis_label='Rating',
            x_range=x_range,
            tools='pan,box_zoom,xwheel_zoom,reset,save'
        )

        if not self.x_selector.init_plot(plot):
            self.plot = figure(title='Failed to init x-axis')
            self.layout = column(self.plot)
            return

        # Benchmark line
        plot.line('x', 'rating', source=self.source,
                  line_width=2, line_color=self.color, legend_label=self.data.tag)

        # Eval run scatter
        plot.scatter('x', 'rating', source=self.eval_source,
                     size=8, color='red', legend_label='Test Run')

        # Store spans so we can update them dynamically
        self.benchmark_spans = []
        self.committee_spans = []

        for gen in self.data.benchmark_gens:
            span = Span(location=0, dimension='height', line_color='orange', line_dash='dashed', line_width=1)
            plot.add_layout(span)
            self.benchmark_spans.append((gen, span))

        for gen in self.data.committee_gens:
            span = Span(location=0, dimension='height', line_color='green', line_dash='dashed', line_width=2)
            plot.add_layout(span)
            self.committee_spans.append((gen, span))

        # Span updater function
        def update_spans(x_var):
            df_indexed = df_benchmark.set_index('mcts_gen')
            for gen, span in self.benchmark_spans:
                if x_var == "mcts_gen":
                    span.location = gen
                elif gen in df_indexed.index and x_var in df_indexed.columns:
                    span.location = df_indexed.at[gen, x_var]
            for gen, span in self.committee_spans:
                if x_var == "mcts_gen":
                    span.location = gen
                elif gen in df_indexed.index and x_var in df_indexed.columns:
                    span.location = df_indexed.at[gen, x_var]

        # Dummy lines for legend
        plot.line(x=[0, 0], y=[0, 0], line_color='orange', line_dash='dashed', line_width=1,
                  legend_label="Evaluated Gens")
        plot.line(x=[0, 0], y=[0, 0], line_color='green', line_dash='dashed', line_width=2,
                  legend_label="Committee")

        plot.legend.location = 'top_left'
        plot.legend.click_policy = 'hide'

        plot.y_range.start = df_benchmark["rating"].min() * 0.9
        plot.y_range.end = max(df_benchmark["rating"].max(), df_eval["rating"].max()) * 1.2

        # Shared control for both data sources
        radio_group = self.x_selector.create_radio_group([plot], [self.source, self.eval_source])

        old_set_x_index = self.x_selector.set_x_index
        def new_set_x_index(x_index, plots, sources, force_refresh=False):
            old_set_x_index(x_index, plots, sources, force_refresh)
            update_spans(self.x_selector.x_column)

        self.x_selector.set_x_index = new_set_x_index
        update_spans(self.x_selector.x_column)

        self.plot = plot
        self.layout = column(plot, row(radio_group))


def create_eval_figure(game: str, tag: str):
    run_params = RunParams(game=game, tag=tag)
    data = EvaluationData(run_params)
    if not data.valid:
        return figure(title='No evaluation data available')

    plotter = EvaluationPlotter(data)
    if not plotter.valid():
        return figure(title='No evaluation data available')

    return plotter.layout
