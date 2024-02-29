"""
A bokeh app to serve the training dashboard.

This is intended to be served within a flask app. The flask app is still a work in progress.

This can be run separately as a standalone app for development purposes. Command:

bokeh serve --show --port 5007 py/alphazero/dashboard/training_app.py --args -g <GAME> -t <TAG>

Within the flask app, you would leave out the --show. You may also need one of the following:

--allow-websocket-origin=127.0.0.1:5000
--allow-websocket-origin=localhost:5007
"""
from alphazero.logic.common_params import CommonParams
from alphazero.logic.directory_organizer import DirectoryOrganizer
from util.sqlite3_util import open_readonly_conn

import argparse
from bokeh.io import curdoc
from bokeh.layouts import gridplot, row
from bokeh.models import BasicTickFormatter, ColumnDataSource, CustomJSTickFormatter, Legend, LegendItem, RadioGroup
from bokeh.palettes import Category10
from bokeh.plotting import figure
import pandas as pd
import sqlite3


HEIGHT = 300
WIDTH = 1200
SIZING_MODE = 'fixed'


def load_args():
    parser = argparse.ArgumentParser()
    CommonParams.add_args(parser)
    return parser.parse_args()


class TrainingVisualizer:
    X_VAR_DICT = {
        'Generation': 'gen',
        'Games': 'n_games',
        'Self-Play Runtime': 'runtime',
        'Num Evaluated Positions': 'n_evaluated_positions',
        'Num Evaluated Batches': 'n_batches_evaluated',
    }

    X_VARS = list(X_VAR_DICT.keys())
    X_VAR_COLUMNS = list(X_VAR_DICT.values())

    def __init__(self, common_params: CommonParams):
        organizer = DirectoryOrganizer(common_params)

        conn = open_readonly_conn(organizer.training_db_filename)
        c = conn.cursor()

        c.execute(
            "SELECT gen, training_start_ts, training_end_ts, window_start, window_end, window_sample_rate FROM training")
        training_data = c.fetchall()

        c.execute(
            "SELECT gen, head_name, loss, loss_weight, accuracy FROM training_heads")
        training_heads_data = c.fetchall()

        # execute cmd to get unique head_names:
        c.execute("SELECT DISTINCT head_name FROM training_heads")
        head_names = c.fetchall()

        conn.close()

        conn = open_readonly_conn(organizer.self_play_db_filename)
        c = conn.cursor()

        c.execute(
            "SELECT gen, positions_evaluated, batches_evaluated, games, augmented_positions FROM self_play_metadata")
        self_play_metadata = c.fetchall()

        c.execute(
            "SELECT gen, client_id, start_timestamp, end_timestamp FROM timestamps")
        timestamp_data = c.fetchall()

        conn.close()

        head_names = [name[0] for name in head_names]

        num_colors = len(head_names)
        assert num_colors <= 10, f'Number of heads ({num_colors}) exceeds the maximum of 10'
        colors = Category10[num_colors]

        self.training_data = training_data
        self.training_heads_data = training_heads_data
        self.self_play_metadata = self_play_metadata
        self.timestamp_data = timestamp_data

        self.head_names = head_names
        self.colors = colors

        self.x_var_index = 0
        self.min_x_dict = {}
        self.max_x_dict = {}

        self.training_df = self.make_training_df()
        self.self_play_metadata_df = self.make_self_play_metadata_df()
        self.shrink_dfs()

        self.loss_weights = self.compute_loss_weights()
        self.source = self.make_source()
        self.compute_min_max_x()
        self.loss_plot = self.make_loss_plot()
        self.stacked_loss_plot = self.make_stacked_loss_plot()
        self.accuracy_plot = self.make_accuracy_plot()
        self.radio_group = self.make_radio_group()
        self.root = self.make_root()

    def shrink_dfs(self):
        gens = set(self.training_df.index)
        self_play_gens = set(self.self_play_metadata_df.index)
        common_gens = list(sorted(gens.intersection(self_play_gens)))
        self.training_df = self.training_df.loc[common_gens]
        self.self_play_metadata_df = self.self_play_metadata_df.loc[common_gens]

    def compute_min_max_x(self):
        for x in TrainingVisualizer.X_VAR_COLUMNS:
            self.min_x_dict[x] = self.source.data[x].iloc[0]
            self.max_x_dict[x] = self.source.data[x].iloc[-1]

    def compute_loss_weights(self):
        head_names = self.head_names
        training_df = self.training_df
        loss_weights = {}
        for name in head_names:
            minimum = training_df[f'loss_weight_{name}'].min()
            maximum = training_df[f'loss_weight_{name}'].max()
            assert minimum == maximum, f'loss_weight_{name} is not constant'
            loss_weights[name] = minimum
        return loss_weights

    def make_training_df(self) -> pd.DataFrame:
        """
        Columns:

        gen (index)
        training_start_ts
        training_end_ts
        window_start
        window_end
        window_sample_rate

        for head_name in head_names:
            {head_name}_loss
            {head_name}_loss_weight
            {head_name}_accuracy
            {head_name}_weighted_loss
        """
        training_data = self.training_data
        training_heads_data = self.training_heads_data

        training_columns = ['gen', 'training_start_ts', 'training_end_ts',
                            'window_start', 'window_end', 'window_sample_rate']
        training_df = pd.DataFrame(training_data, columns=training_columns)
        # training_df.set_index('gen', inplace=True)

        # Convert training_heads_data into a DataFrame
        training_heads_columns = ['gen', 'head_name',
                                  'loss', 'loss_weight', 'accuracy']
        training_heads_df = pd.DataFrame(
            training_heads_data, columns=training_heads_columns)
        training_heads_df['weighted_loss'] = training_heads_df['loss'] * \
            training_heads_df['loss_weight']

        # Pivot the DataFrame so that each unique head name becomes a column
        training_heads_df = training_heads_df.pivot(
            index='gen', columns='head_name')
        training_heads_df = training_heads_df.dropna()

        # Flatten the MultiIndex columns and join them with an underscore
        training_heads_df.columns = [
            '_'.join(col).strip() for col in training_heads_df.columns.values]

        # Merge training_heads_df with training_df
        training_df = pd.merge(
            training_df, training_heads_df, on='gen', how='inner')

        return training_df

    def make_self_play_metadata_df(self) -> pd.DataFrame:
        self_play_metadata = self.self_play_metadata
        timestamp_data = self.timestamp_data

        self_play_metadata_columns = ['gen', 'positions_evaluated',
                                      'batches_evaluated', 'games', 'augmented_positions']
        self_play_metadata_df = pd.DataFrame(
            self_play_metadata, columns=self_play_metadata_columns)
        # self_play_metadata_df.set_index('gen', inplace=True)

        timestamp_columns = ['gen', 'client_id', 'start_timestamp', 'end_timestamp']
        timestamp_df = pd.DataFrame(timestamp_data, columns=timestamp_columns)
        timestamp_df.set_index('gen', inplace=True)

        total_time = timestamp_df['end_timestamp'] - timestamp_df['start_timestamp']
        total_time = total_time.groupby('gen').sum()
        total_time *= 1e-9
        self_play_metadata_df['total_time'] = total_time

        return self_play_metadata_df

    def make_source(self):
        training_df = self.training_df
        self_play_metadata_df = self.self_play_metadata_df
        head_names = self.head_names

        data = {
            'gen': training_df['gen'],
            'n_games': self_play_metadata_df['games'].cumsum(),
            'n_evaluated_positions': self_play_metadata_df['positions_evaluated'].cumsum(),
            'n_batches_evaluated': self_play_metadata_df['batches_evaluated'].cumsum(),
            'runtime': self_play_metadata_df['total_time'].cumsum(),
        }
        for i, name in enumerate(head_names):
            data[f'loss_{name}'] = training_df[f'loss_{name}']
            data[f'weighted_loss_{name}'] = training_df[f'weighted_loss_{name}']
            data[f'accuracy_{name}'] = training_df[f'accuracy_{name}']
            if i == 0:
                data[f'cumulative_weighted_loss_{name}'] = training_df[f'weighted_loss_{name}']
            else:
                data[f'cumulative_weighted_loss_{name}'] = data[f'cumulative_weighted_loss_{head_names[i-1]}'] + \
                    training_df[f'weighted_loss_{name}']

        cls = TrainingVisualizer
        data['x'] = data[cls.X_VAR_COLUMNS[self.x_var_index]]

        return ColumnDataSource(data=data)

    def make_radio_group(self):
        cls = TrainingVisualizer
        radio_group = RadioGroup(labels=cls.X_VARS, active=self.x_var_index)

        x_range = self.loss_plot.x_range
        plots = [self.loss_plot, self.stacked_loss_plot]

        def update_data(attr, old, new):
            prev_x_var_index = self.x_var_index
            self.x_var_index = radio_group.active
            x_var_column = cls.X_VAR_COLUMNS[self.x_var_index]
            self.source.data['x'] = self.source.data[x_var_column]

            for plot in plots:
                plot.xaxis.axis_label = cls.X_VARS[self.x_var_index]

                if x_var_column == 'runtime':
                    plot.xaxis.formatter = CustomJSTickFormatter(code="""
                        var total_seconds = tick;
                        var days = Math.floor(total_seconds / 86400);
                        var hours = Math.floor((total_seconds % 86400) / 3600);
                        var minutes = Math.floor((total_seconds % 3600) / 60);
                        if (days > 0) {
                            return days + "d " + hours + "h " + minutes + "m";
                        } else {
                            return hours + "h " + minutes + "m";
                        }
                    """)
                else:
                    plot.xaxis.formatter = BasicTickFormatter()

            if self.x_var_index != prev_x_var_index:
                prev_x_var_column = cls.X_VAR_COLUMNS[prev_x_var_index]
                start = x_range.start
                end = x_range.end
                prev_x_min = self.min_x_dict[prev_x_var_column]
                prev_x_max = self.max_x_dict[prev_x_var_column]
                prev_x_width = prev_x_max - prev_x_min
                if prev_x_width > 0:
                    start_pct = (start - prev_x_min) / prev_x_width
                    end_pct = (end - prev_x_min) / prev_x_width
                    x_min = self.min_x_dict[x_var_column]
                    x_max = self.max_x_dict[x_var_column]
                    x_width = x_max - x_min
                    x_range.start = x_min + start_pct * x_width
                    x_range.end = x_min + end_pct * x_width

        radio_group.on_change('active', update_data)
        return radio_group

    def make_root(self):
        grid = gridplot([[self.loss_plot], [self.stacked_loss_plot], [self.accuracy_plot]],
                        toolbar_location='above')

        root = row([self.radio_group, grid])

        return root

    def make_loss_plot(self):
        head_names = self.head_names
        colors = self.colors

        x_axis_label = TrainingVisualizer.X_VARS[self.x_var_index]
        plot = figure(
            height=HEIGHT, width=WIDTH, sizing_mode=SIZING_MODE,
            title='Train Loss', x_axis_label=x_axis_label,
            y_axis_label='Loss', tools='pan,box_zoom,xwheel_zoom,reset,save')

        # Add lines to the plot and create legend items
        legend_items = []
        for i, (name, color) in enumerate(zip(head_names, colors)):
            line_width = 2 if i < 2 else 1
            line = plot.line('x', f'loss_{name}', source=self.source,
                            line_width=line_width, color=color)

            legend_item = LegendItem(label=name, renderers=[line])
            legend_items.append(legend_item)

        legend_items.reverse()
        legend = Legend(items=legend_items, click_policy='hide')
        plot.add_layout(legend, 'right')
        return plot

    def make_stacked_loss_plot(self):
        head_names = self.head_names
        colors = self.colors

        x_axis_label = TrainingVisualizer.X_VARS[self.x_var_index]
        plot = figure(
            height=HEIGHT, width=WIDTH, sizing_mode=SIZING_MODE,
            title='Stacked Train Loss', x_axis_label=x_axis_label,
            y_axis_label='Loss', x_range=self.loss_plot.x_range,
            tools='pan,box_zoom,xwheel_zoom,reset,save')

        legend_items = []
        # Create the stacked area plot
        for i, name in enumerate(head_names):
            color = colors[i]
            y2 = f'cumulative_weighted_loss_{name}'
            fill_alpha = 1 if i < 2 else 0.5
            if i == 0:
                plot.varea(x='x', y1=0, y2=y2, color=color, fill_alpha=fill_alpha, source=self.source)
            else:
                y1 = f'cumulative_weighted_loss_{head_names[i-1]}'
                plot.varea(x='x', y1=y1, y2=y2, color=color, fill_alpha=fill_alpha, source=self.source)

            weight = self.loss_weights[name]
            if weight == 1:
                weighted_name = name
            else:
                weighted_name = '%g * %s' % (self.loss_weights[name], name)
            legend_item = LegendItem(label=weighted_name, renderers=[plot.renderers[i]])
            legend_items.append(legend_item)

        legend_items.reverse()
        legend = Legend(items=legend_items)
        plot.add_layout(legend, 'right')

        return plot

    def make_accuracy_plot(self):
        head_names = self.head_names
        colors = self.colors

        x_axis_label = TrainingVisualizer.X_VARS[self.x_var_index]
        accuracy_plot = figure(
            height=HEIGHT, width=WIDTH, sizing_mode=SIZING_MODE,
            title='Train Accuracy', x_axis_label=x_axis_label,
            y_axis_label='Accuracy', x_range=self.loss_plot.x_range, y_range=[0, 1],
            tools='pan,box_zoom,xwheel_zoom,reset,save')
        legend_items = []
        for i, (name, color) in enumerate(zip(head_names, colors)):
            line_width = 2 if i < 2 else 1
            y = f'accuracy_{name}'
            line = accuracy_plot.line(
                x='x', y=y, line_width=line_width, color=color, source=self.source)
            legend_items.append((name, [line]))

        legend_items.reverse()
        legend = Legend(items=legend_items)
        legend.click_policy = 'hide'
        accuracy_plot.add_layout(legend, 'right')
        return accuracy_plot


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    viz = TrainingVisualizer(common_params)
    curdoc().title = f'{common_params.game} {common_params.tag} Dashboard'
    curdoc().add_root(viz.root)


main()
