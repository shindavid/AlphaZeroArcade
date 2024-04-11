from alphazero.dashboard.ratings_figure import create_ratings_figure
from alphazero.logic.run_params import RunParams

from bokeh.embed import server_document
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from flask import Flask, jsonify, render_template, request, session
from tornado.ioloop import IOLoop

import argparse
from dataclasses import dataclass
import os
import secrets
from threading import Thread
from typing import List


@dataclass
class Params:
    bokeh_port: int = 5012
    flask_port: int = 8002
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
        group = parser.add_argument_group('Flaskh/Bokeh options')

        defaults = Params()
        group.add_argument('--bokeh-port', type=int, default=defaults.bokeh_port,
                           help='bokeh port (default: %(default)s)')
        group.add_argument('--flask-port', type=int, default=defaults.flask_port,
                           help='flask port (default: %(default)s)')
        group.add_argument('--debug', action='store_true', help='debug mode')



parser = argparse.ArgumentParser()

RunParams.add_args(parser, multiple_tags=True)
Params.add_args(parser)

args = parser.parse_args()
run_params = RunParams.create(args, require_tag=False)
params = Params.create(args)

bokeh_port = params.bokeh_port
flask_port = params.flask_port
theme = Theme(filename="static/theme.yaml")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
if params.debug:
    app.debug = True

game_dir = os.path.join(run_params.output_dir, run_params.game)
if not os.path.isdir(game_dir):
    raise ValueError(f'Directory does not exist: {game_dir}')

all_tags = [d for d in os.listdir(game_dir) if os.path.isdir(os.path.join(game_dir, d))]
if not all_tags:
    raise ValueError(f'No directories found in {game_dir}')

if run_params.tag:
    tags = run_params.tag.split(',')
    for tag in tags:
        if not tag:
            raise ValueError(f'Bad --tag/-t argument: {run_params.tag}')
        path = os.path.join(run_params.output_dir, run_params.game, tag)
        if not os.path.isdir(path):
            raise ValueError(f'Directory does not exist: {path}')
else:
    tags = all_tags

default_tags = tags


def training_policy(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    if not tag_str:
        return
    tags = tag_str.split(',')

    plot = figure(title=f"Training - Policy ({tag_str})")
    plot.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_color="blue")
    doc.add_root(plot)
    doc.theme = theme


def training_value(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    if not tag_str:
        return
    tags = tag_str.split(',')

    plot = figure(title=f"Training - Value ({tag_str})")
    plot.line([1, 2, 3, 4, 5], [6, 9, 2, 1, 5], line_color="blue")
    doc.add_root(plot)
    doc.theme = theme


def training_combined(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    if not tag_str:
        return
    tags = tag_str.split(',')

    plot = figure(title=f"Training - Combined ({tag_str})")
    plot.line([1, 2, 3, 4, 5], [6, 3, 2, 3, 3], line_color="blue")
    doc.add_root(plot)
    doc.theme = theme


def self_play(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    if not tag_str:
        return
    tags = tag_str.split(',')

    plot = figure(title=f"Self-Play ({tag_str})")
    plot.line([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], line_color="red")
    doc.add_root(plot)
    doc.theme = theme


def ratings(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    doc.add_root(create_ratings_figure(run_params.output_dir, run_params.game, tags))
    doc.theme = theme


class DocumentCollection:
    def __init__(self, tags: List[str]):
        tag_str = ','.join(tags)

        training_heads = ['policy', 'value', 'combined']

        training_data = [(head, server_document(f'http://localhost:{bokeh_port}/training_{head}',
                                                arguments={'tags': tag_str}))
                        for head in training_heads]
        self_play = server_document(f'http://localhost:{bokeh_port}/self_play',
                                    arguments={'tags': tag_str})
        ratings = server_document(f'http://localhost:{bokeh_port}/ratings',
                                arguments={'tags': tag_str})

        self.tags = tags
        self.training_data = training_data
        self.self_play = self_play
        self.ratings = ratings

    def get_base_data(self):
        return {
            'training_data': self.training_data,
            'self_play': self.self_play,
            'ratings': self.ratings,
            'tags': all_tags,
            'init_tags': self.tags,
        }

    def get_update_data(self):
        return {
            'training_policy': self.training_data[0][1],
            'training_value': self.training_data[1][1],
            'training_combined': self.training_data[2][1],
            'self_play': self.self_play,
            'ratings': self.ratings,
        }


@app.route('/', methods=['GET'])
def bkapp_page():
    # Get the tags from the session
    tags = session.get('tags', default_tags)
    docs = DocumentCollection(tags)
    data = docs.get_base_data()
    return render_template("dashboard.html", template="Flask", **data)


@app.route('/update_plots', methods=['POST'])
def update_plots():
    form_lists = list(request.form.lists())
    if not form_lists:
        tags = []
    else:
        tags = form_lists[0][1]  # I don't get it, but this works
    session['tags'] = tags
    docs = DocumentCollection(tags)
    data = docs.get_update_data()
    return jsonify(data)


def bk_worker():
    apps = {
        '/training_policy': training_policy,
        '/training_value': training_value,
        '/training_combined': training_combined,
        '/self_play': self_play,
        '/ratings': ratings,
    }

    allow_list = [
        f"localhost:{bokeh_port}",
        f"127.0.0.1:{flask_port}",
        f"localhost:{flask_port}"]

    server = Server(apps, io_loop=IOLoop(),
                    allow_websocket_origin=allow_list,
                    port=bokeh_port)

    server.start()
    server.io_loop.start()


def main():
    Thread(target=bk_worker).start()
    app.run(port=flask_port)


if __name__ == '__main__':
    main()
