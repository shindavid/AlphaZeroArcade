"""
A bokeh app to serve the training dashboard.

This is intended to be served within a flask app. The flask app is still a work in progress.

This can be run separately as a standalone app for development purposes. Command:

bokeh serve --show --port 5007 py/alphazero/dashboard/training_app.py --args -g <GAME> -t <TAG>

Within the flask app, you would leave out the --show. You may also need one of the following:

--allow-websocket-origin=127.0.0.1:5000
--allow-websocket-origin=localhost:5007
"""
from .x_var_logic import XVarSelector, make_x_df
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import bokeh_util

from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Legend, LegendItem, Select, Span
from bokeh.plotting import figure
import pandas as pd

import sqlite3
from typing import Dict, List, Optional


Tag = str


def create_training_figure(game: str, tags: List[Tag], head: str):
    head_data_list: List[HeadData] = []
    for tag in tags:
        run_params = RunParams(game, tag)
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        conn = sqlite3.connect(organizer.training_db_filename)
        head_data = HeadData(conn, tag, head)
        conn.close()
        if not head_data.valid:
            continue

        x_df = make_x_df(organizer)
        head_data.join(x_df)
        head_data_list.append(head_data)

    if not head_data_list:
        return figure(title='No data available')

    plotter = TrainingPlotter(head, head_data_list)
    return plotter.figure


def create_combined_training_figure(game: str, tags: List[Tag]):
    head_data_dict: Dict[Tag, HeadData] = {}
    for tag in tags:
        run_params = RunParams(game, tag)
        organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')

        conn = sqlite3.connect(organizer.training_db_filename)
        head_data = HeadData(conn, tag, None)
        conn.close()

        if not head_data.valid:
            continue

        x_df = make_x_df(organizer)
        head_data.join(x_df)
        head_data_dict[head_data.tag] = head_data

    if not head_data_dict:
        return figure(title='No data available')

    plotter = CombinedTrainingPlotter(head_data_dict)
    return plotter.figure


class HeadData:
    def __init__(self, conn: sqlite3.Connection, tag: Tag, head: Optional[str]):
        self.tag = tag
        columns = ['head_name', 'gen', 'loss', 'loss_weight']
        self.head = head
        c = conn.cursor()
        if head is None:
            query = f"SELECT {', '.join(columns)} FROM training_heads"
            c.execute(query)
        else:
            query = f"SELECT {', '.join(columns)} FROM training_heads WHERE head_name = ?"
            c.execute(query, (head, ))
        head_data = c.fetchall()

        self.valid = len(head_data) > 0
        if not self.valid:
            return

        df = pd.DataFrame(head_data, columns=columns)

        self.head_names = list(df['head_name'].unique())
        df['weighted_loss'] = df['loss'] * df['loss_weight']

        # Pivot the DataFrame so that each unique head name becomes a column
        df = df.pivot(index='gen', columns='head_name')
        df = df.dropna()

        # Flatten the MultiIndex columns and join them with an underscore
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

        self.df = df

        self.loss_weights = self.compute_loss_weights()

    def compute_loss_weights(self):
        head_names = self.head_names
        df = self.df
        loss_weights = {}
        for name in head_names:
            minimum = df[f'loss_weight_{name}'].min()
            maximum = df[f'loss_weight_{name}'].max()
            assert minimum == maximum, f'loss_weight_{name} is not constant'
            loss_weights[name] = minimum

        return loss_weights

    def join(self, x_df: pd.DataFrame):
        x_df = x_df.reset_index()
        self.df = pd.merge(self.df, x_df, left_on='gen', right_on='mcts_gen',
                           how='inner').reset_index()


class TrainingPlotter:
    def __init__(self, head: str, head_data_list: List[HeadData]):
        self.head = head
        self.colors = bokeh_util.get_colors(len(head_data_list))
        self.head_data_list = head_data_list

        self.x_var_selector = XVarSelector([hd.df for hd in head_data_list])
        self.sources: Dict[str, ColumnDataSource] = {}

        self.load()
        self.loss_plot = self.make_loss_plot()
        self.figure = self.make_figure()

    def load(self):
        for head_data in self.head_data_list:
            source = ColumnDataSource(head_data.df)
            self.x_var_selector.init_source(source)
            self.sources[head_data.tag] = source

    def make_figure(self):
        loss_plot = self.loss_plot
        plots = [loss_plot]
        sources = list(self.sources.values())
        radio_group = self.x_var_selector.create_radio_group(plots, sources)
        return column(loss_plot, radio_group)

    def make_loss_plot(self):
        head = self.head
        colors = self.colors
        y = f'loss_{head}'

        plot = figure(
            title=f'Train Loss - {head}', x_range=[0, 1], y_axis_label='Loss',
            active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
        self.x_var_selector.init_plot(plot)

        for head_data, color in zip(self.head_data_list, colors):
            label = head_data.tag
            source = self.sources[label]
            plot.line('x', y, source=source, line_color=color, legend_label=label)

        hline = Span(location=0, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)
        plot.y_range.start = 0

        plot.legend.location = 'bottom_right'
        plot.legend.click_policy = 'hide'
        return plot


class CombinedTrainingPlotter:
    def __init__(self, head_data_dict: Dict[Tag, HeadData]):
        self.head_data_dict = head_data_dict

        self.x_var_selector = XVarSelector([hd.df for hd in head_data_dict.values()])
        self.sources: Dict[str, ColumnDataSource] = {}

        self.load()
        self.combined_plot = self.make_combined_loss_plot()
        self.figure = self.make_figure()

    def load(self):
        for head_data in self.head_data_dict.values():
            source = ColumnDataSource(head_data.df)
            data = source.data

            combined_loss = None
            for i, name in enumerate(head_data.head_names):
                if i == 0:
                    var = f'weighted_loss_{name}'
                    combined_loss = data[var]
                    data[f'cumulative_weighted_loss_{name}'] = combined_loss
                else:
                    var = f'weighted_loss_{name}'
                    combined_loss = combined_loss + data[var]
                    data[f'cumulative_weighted_loss_{name}'] = combined_loss

            data['combined_loss'] = combined_loss
            self.x_var_selector.init_source(source)
            self.sources[head_data.tag] = source

    def make_figure(self):
        tags = list(self.head_data_dict.keys())
        select = Select(title="Select Tag", value=tags[0], options=tags)
        select_wrapper = column(select, sizing_mode="fixed")

        initial_plot = self.make_stacked_loss_plot(self.head_data_dict[select.value])

        def update_plot(attr, old, new):
            new_plot = self.make_stacked_loss_plot(self.head_data_dict[select.value])
            new_layout = gridplot([
                [None, select_wrapper],
                [self.combined_plot, new_plot],
                [radio_group, None]
            ], sizing_mode='scale_height')
            layout.children[:] = new_layout.children

        select.on_change('value', update_plot)

        plots = [self.combined_plot] + [initial_plot]
        sources = list(self.sources.values())
        radio_group = self.x_var_selector.create_radio_group(plots, sources)

        layout = gridplot([
            [None, select_wrapper],
            [self.combined_plot, initial_plot],
            [radio_group, None]
            ], sizing_mode='scale_height')

        return layout

    def make_combined_loss_plot(self):
        colors = bokeh_util.get_colors(len(self.head_data_dict))
        y = 'combined_loss'

        plot = figure(
            title=f'Train Loss - combined', x_range=[0, 1], y_axis_label='Loss',
            active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
        self.x_var_selector.init_plot(plot)

        for head_data, color in zip(self.head_data_dict.values(), colors):
            label = head_data.tag
            source = self.sources[label]
            plot.line('x', y, source=source, line_color=color, legend_label=label)

        hline = Span(location=0, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)
        plot.y_range.start = 0

        plot.legend.location = 'top_right'
        plot.legend.click_policy = 'hide'
        return plot

    def make_stacked_loss_plot(self, head_data: HeadData):
        loss_weights = head_data.loss_weights
        source = self.sources[head_data.tag]

        head_names = head_data.head_names
        colors = bokeh_util.get_colors(len(head_names))

        title = f'Stacked Train Loss ({head_data.tag})'
        plot = figure(
            title=title, y_axis_label='Loss', x_range=[0, 1],
            active_scroll='xwheel_zoom', tools='pan,box_zoom,xwheel_zoom,reset,save')
        self.x_var_selector.init_plot(plot)

        legend_items = []
        # Create the stacked area plot
        for i, name in enumerate(head_names):
            color = colors[i]
            y2 = f'cumulative_weighted_loss_{name}'
            fill_alpha = 1 if i < 2 else 0.5
            if i == 0:
                plot.varea(x='x', y1=0, y2=y2, color=color,
                           fill_alpha=fill_alpha, source=source)
            else:
                y1 = f'cumulative_weighted_loss_{head_names[i-1]}'
                plot.varea(x='x', y1=y1, y2=y2, color=color,
                           fill_alpha=fill_alpha, source=source)

            weight = loss_weights[name]
            if weight == 1:
                weighted_name = name
            else:
                weighted_name = '%g * %s' % (weight, name)
            legend_item = LegendItem(label=weighted_name, renderers=[plot.renderers[i]])
            legend_items.append(legend_item)

        hline = Span(location=0, dimension='width', line_color='gray',
                     line_dash='dashed', line_width=1)
        plot.add_layout(hline)
        plot.y_range.start = 0

        legend_items.reverse()
        legend = Legend(items=legend_items)
        plot.add_layout(legend)
        plot.legend.location = 'top_right'

        return plot
