#!/usr/bin/env python3
from alphazero.dashboard.training_plotting import create_training_figure, \
    create_combined_training_figure
from alphazero.dashboard.rating_plotting import create_ratings_figure
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

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
import sqlite3
from threading import Thread
import time
from typing import List
import webbrowser


@dataclass
class Params:
    bokeh_port: int = 5012
    flask_port: int = 8002
    open_in_browser: bool = True
    debug: bool = False

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            bokeh_port=args.bokeh_port,
            flask_port=args.flask_port,
            open_in_browser=args.open_in_browser,
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
        group.add_argument('-o', '--open-in-browser', action='store_true',
                            help='automatically open dashboard in web browser')
        group.add_argument('-d', '--debug', action='store_true', help='debug mode')


parser = argparse.ArgumentParser()

RunParams.add_args(parser, multiple_tags=True)
Params.add_args(parser)

args = parser.parse_args()
run_params = RunParams.create(args, require_tag=False)
params = Params.create(args)

bokeh_port = params.bokeh_port
flask_port = params.flask_port

cur_dir = os.path.dirname(os.path.abspath(__file__))
theme = Theme(filename=os.path.join(cur_dir, "static/theme.yaml"))

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


all_training_heads = []
for tag in all_tags:
    rp = RunParams(run_params.output_dir, run_params.game, tag)
    directory_organizer = DirectoryOrganizer(rp)
    training_db_filename = directory_organizer.training_db_filename
    if not os.path.isfile(training_db_filename):
        continue

    conn = sqlite3.connect(training_db_filename)
    c = conn.cursor()

    # get unique head_name values from table training_heads:
    c.execute('SELECT DISTINCT head_name FROM training_heads')
    heads = [r[0] for r in c.fetchall()]

    for h in heads:
        if h not in all_training_heads:
            all_training_heads.append(h)

    conn.close()


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


def training_head(head: str):
    def training_inner(doc):
        tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
        tags = [t for t in tag_str.split(',') if t]
        doc.add_root(create_training_figure(run_params.output_dir, run_params.game, tags, head))
        doc.theme = theme

    return training_inner


def training(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    doc.add_root(create_combined_training_figure(run_params.output_dir, run_params.game, tags))
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

        training_heads = [
            (head, server_document(f'http://localhost:{bokeh_port}/training_{head}',
                                   arguments={'tags': tag_str}))
                                   for head in all_training_heads]
        training = server_document(f'http://localhost:{bokeh_port}/training',
                                            arguments={'tags': tag_str})
        self_play = server_document(f'http://localhost:{bokeh_port}/self_play',
                                    arguments={'tags': tag_str})
        ratings = server_document(f'http://localhost:{bokeh_port}/ratings',
                                arguments={'tags': tag_str})

        self.tags = tags
        self.training_heads = training_heads
        self.training = training
        self.self_play = self_play
        self.ratings = ratings

    def get_base_data(self):
        return {
            'training_heads': self.training_heads,
            'training': self.training,
            'self_play': self.self_play,
            'ratings': self.ratings,
            'tags': all_tags,
            'init_tags': self.tags,
        }

    def get_update_data(self):
        d = {
            'training': self.training,
            'self_play': self.self_play,
            'ratings': self.ratings,
            }
        for h, head in enumerate(all_training_heads):
            d[f'training_{head}'] = self.training_heads[h][1]
        return d


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
        '/training': training,
        '/self_play': self_play,
        '/ratings': ratings,
    }

    for head in all_training_heads:
        apps[f'/training_{head}'] = training_head(head)

    allow_list = [
        f"localhost:{bokeh_port}",
        f"127.0.0.1:{flask_port}",
        f"localhost:{flask_port}"]

    server = Server(apps, io_loop=IOLoop(),
                    allow_websocket_origin=allow_list,
                    port=bokeh_port)

    server.start()
    server.io_loop.start()


def open_in_browser():
    time.sleep(0.2)
    print('Opening dashboard in web browser...')
    webbrowser.open(f'http://127.0.0.1:{flask_port}/')


def main():
    Thread(target=bk_worker).start()
    if params.open_in_browser:
        Thread(target=open_in_browser).start()
    else:
        print('*********************************************************')
        print('To view the dashboard, open a web browser and navigate to')
        print(f'http://127.0.0.1:{flask_port}/')
        print('*********************************************************')
        print('')

    app.run(port=flask_port)


if __name__ == '__main__':
    main()
