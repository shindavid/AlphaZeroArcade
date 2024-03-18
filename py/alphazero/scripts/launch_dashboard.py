#!/usr/bin/env python3

from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.repo_util import Repo

from bokeh.embed import components, server_document
from bokeh.io import curdoc
from bokeh.layouts import gridplot, layout
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Category10
from bokeh.plotting import figure
from flask import Flask, render_template, request, make_response
import pandas as pd

import argparse
from dataclasses import dataclass
import os
import pipes
import subprocess


app = Flask(__name__)


@dataclass
class Params:
    bokeh_port: int = 5007
    flask_port: int = 5000
    debug: bool = False

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            bokeh_port=args.bokeh_port,
            flask_port=args.flask_port,
            debug=bool(args.debug),
            )

    @staticmethod
    def add_args(parser):
        defaults = Params()
        parser.add_argument('--bokeh-port', type=int, default=defaults.bokeh_port,
                            help='bokeh port (default: %(default)s)')
        parser.add_argument('--flask-port', type=int, default=defaults.flask_port,
                            help='flask port (default: %(default)s)')
        parser.add_argument('--debug', action='store_true', help='debug mode')


def load_args():
    parser = argparse.ArgumentParser()

    RunParams.add_args(parser)
    Params.add_args(parser)

    return parser.parse_args()


GLOBALS = {}


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    run_params = GLOBALS['run_params']
    params = GLOBALS['params']

    # training_db_filename = organizer.training_db_filename
    log_file = request.form.get('log_file')

    # database_data = DatabaseData(training_db_filename)
    # training_data = database_data.training_data

    # training_plot = database_data.make_training_plot()
    # training_script, training_div = components(training_plot)

    # self_play_plot = database_data.make_training_plot()
    # self_play_script, self_play_div = components(self_play_plot)

    # Get the last 10 lines of the log file
    if log_file:
        with open(log_file, 'r') as f:
            logs = f.readlines()[-10:]
    else:
        logs = []

    title = f'{run_params.game} {run_params.tag} Dashboard'

    bokeh_server_url = f'http://localhost:{params.bokeh_port}/training_app'
    script = server_document(bokeh_server_url)
    # script = server_document("http://localhost:5009/bokeh_app")

    # Pass the plot components and data to the template
    return render_template('dashboard.html',
                            title=title,
                            script=script,
                            logs=logs)


def main():
    args = load_args()
    run_params = RunParams.create(args)
    params = Params.create(args)
    GLOBALS['params'] = params
    GLOBALS['run_params'] = run_params

    os.chdir(Repo.root())
    bokeh_cmd = ['bokeh', 'serve',
                 f'--allow-websocket-origin=localhost:{params.bokeh_port}',
                 f'--allow-websocket-origin=127.0.0.1:{params.flask_port}',
                 '--port', str(params.bokeh_port),
                 'py/alphazero/dashboard/training_app.py', '--args',
                 ]
    run_params.add_to_cmd(bokeh_cmd)
    print(' '.join(map(pipes.quote, bokeh_cmd)))
    p = subprocess.Popen(bokeh_cmd)
    try:
        app.run(debug=params.debug, port=params.flask_port)
    finally:
        p.kill()


if __name__ == '__main__':
    main()
