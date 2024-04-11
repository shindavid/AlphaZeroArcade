from bokeh.embed import server_document
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from flask import Flask, jsonify, render_template, request, session
from tornado.ioloop import IOLoop

import argparse
from dataclasses import dataclass
import secrets
from threading import Thread
from typing import List


@dataclass
class Params:
    bokeh_port: int = 5012
    flask_port: int = 8002
    tag_str: str = ''
    debug: bool = False

    @property
    def tags(self) -> List[str]:
        return self.tag_str.split(',')

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            bokeh_port=args.bokeh_port,
            flask_port=args.flask_port,
            tag_str=args.tags,
            debug=bool(args.debug),
        )

    @staticmethod
    def add_args(parser):
        defaults = Params()
        parser.add_argument('--bokeh-port', type=int, default=defaults.bokeh_port,
                            help='bokeh port (default: %(default)s)')
        parser.add_argument('--flask-port', type=int, default=defaults.flask_port,
                            help='flask port (default: %(default)s)')
        parser.add_argument('-t', '--tags', default=defaults.tag_str,
                            help='comma-separated list of tags')
        parser.add_argument('--debug', action='store_true', help='debug mode')



parser = argparse.ArgumentParser()
Params.add_args(parser)
params = Params.create(parser.parse_args())

bokeh_port = params.bokeh_port
flask_port = params.flask_port
theme = Theme(filename="static/theme.yaml")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
if params.debug:
    app.debug = True


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
    if not tag_str:
        return
    tags = tag_str.split(',')

    plot = figure(title=f"Ratings ({tag_str})")
    plot.line([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], line_color="green")
    doc.add_root(plot)
    doc.theme = theme


class Documents:
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
            'tags': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
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
    tags = session.get('tags', params.tags)
    docs = Documents(tags)
    data = docs.get_base_data()
    return render_template("dashboard.html", template="Flask", **data)


@app.route('/update_plots', methods=['POST'])
def update_plots():
    tags = list(request.form.lists())[0][1]  # I don't get it, but this works
    session['tags'] = tags
    docs = Documents(tags)
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


Thread(target=bk_worker).start()

if __name__ == '__main__':
    app.run(port=flask_port)
