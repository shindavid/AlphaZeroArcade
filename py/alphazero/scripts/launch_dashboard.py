#!/usr/bin/env python3
from alphazero.dashboard.eval_plotting import create_eval_figure
from alphazero.dashboard.training_plotting import create_training_figure, \
    create_combined_training_figure
from alphazero.dashboard.rating_plotting import create_ratings_figure
from alphazero.dashboard.self_play_plotting import create_self_play_figure
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter
from util.str_util import rreplace

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import Select
from bokeh.server.server import Server
from bokeh.themes import Theme
from flask import Flask, jsonify, render_template, request, session
from tornado.ioloop import IOLoop

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
import secrets
import sqlite3
import sys
from threading import Thread
from typing import Dict, List


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
        group = parser.add_argument_group('Flask/Bokeh options')

        defaults = Params()
        group.add_argument('--bokeh-port', type=int, default=defaults.bokeh_port,
                           help='bokeh port (default: %(default)s)')
        group.add_argument('--flask-port', type=int, default=defaults.flask_port,
                           help='flask port (default: %(default)s)')
        group.add_argument('-d', '--debug', action='store_true', help='debug mode')


parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

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

game_dir = os.path.join('/workspace/output', run_params.game)
if not os.path.isdir(game_dir):
    raise ValueError(f'Directory does not exist: {game_dir}')

all_tags = [d for d in os.listdir(game_dir) if os.path.isdir(os.path.join(game_dir, d))]
if not all_tags:
    raise ValueError(f'No directories found in {game_dir}')

usable_tags = []
skipped_tags = []
for tag in all_tags:
    rp = RunParams(run_params.game, tag)
    directory_organizer = DirectoryOrganizer(rp, base_dir_root='/workspace')
    if directory_organizer.version_check():
        usable_tags.append(tag)
    else:
        skipped_tags.append(tag)

if skipped_tags:
    print('The following tags were skipped because of a version mismatch:')
    for tag in skipped_tags:
        print(f'  {tag}')
    print('')
    print('This means that those output dirs were created with an outdated version of the code.')
    print('Consider removing them.')

if all_tags and not usable_tags:
    print('All output directories are outdated. Exiting...')
    sys.exit(0)

# sort usable_tags based on os.path.getmtime:
usable_tags.sort(key=lambda x: os.path.getmtime(os.path.join(game_dir, x)))

all_training_heads = []
for tag in usable_tags:
    rp = RunParams(run_params.game, tag)
    directory_organizer = DirectoryOrganizer(rp, base_dir_root='/workspace')
    training_db_filename = directory_organizer.training_db_filename
    if not os.path.isfile(training_db_filename):
        continue

    conn = sqlite3.connect(training_db_filename)
    c = conn.cursor()

    c.execute('SELECT head_name FROM training_heads WHERE gen = 1')
    heads = [r[0] for r in c.fetchall()]

    for h in heads:
        if h not in all_training_heads:
            all_training_heads.append(h)

    conn.close()

if run_params.tag:
    tags = run_params.tag.split(',')
    outdated_tags = []
    for tag in tags:
        if not tag:
            raise ValueError(f'Bad --tag/-t argument: {run_params.tag}')
        path = os.path.join('/workspace/output', run_params.game, tag)
        if not os.path.isdir(path):
            raise ValueError(f'Directory does not exist: {path}')
        rp = RunParams(run_params.game, tag)
        directory_organizer = DirectoryOrganizer(rp, base_dir_root='/workspace')
        if not directory_organizer.version_check():
            outdated_tags.append(tag)
    if outdated_tags:
        print(f'ERROR: The following tags for {run_params.game} are outdated:')
        for tag in outdated_tags:
            print(f'  {tag}')
        print('')
        print('This means that the output dir was created with an outdated version of the code.')
        print('Please remove the outdated tags and try again.')
        sys.exit(0)
else:
    tags = usable_tags

default_tags = tags


def get_benchmark_tags(tag: str) -> List[str]:
    """
    returns a list of benchmark tags used for evaluating the given run tag
    """
    rp = RunParams(run_params.game, tag)
    directory_organizer = DirectoryOrganizer(rp, base_dir_root='/workspace')

    benchmark_tags = []
    filename = directory_organizer.benchmark_db_filename
    if os.path.isfile(filename):
        benchmark_tags.append(tag)

    eval_dir = directory_organizer.eval_db_dir
    if os.path.exists(eval_dir):
        for f in os.listdir(eval_dir):
            if f.endswith('.db'):
                benchmark_tags.append(os.path.splitext(f)[0])
    return benchmark_tags

def get_benchmark_eval_mapping(tags: List[str]) -> Dict[str, List[str]]:
    """
    returns a dict mapping a benchmark tag to a list of run tags that are evaluated against it.
    """
    tag_dict = {tag: get_benchmark_tags(tag) for tag in tags}
    benchmark_dict = defaultdict(list)
    for tag, benchmark_tags in tag_dict.items():
        for benchmark_tag in benchmark_tags:
            benchmark_dict[benchmark_tag].append(tag)
    return benchmark_dict


def training_head(head: str):
    def training_inner(doc):
        tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
        tags = [t for t in tag_str.split(',') if t]
        doc.add_root(create_training_figure(run_params.game, tags, head))
        doc.theme = theme

    return training_inner


def training(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    doc.add_root(create_combined_training_figure(run_params.game, tags))
    doc.theme = theme


def self_play(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    doc.add_root(create_self_play_figure(run_params.game, tags))
    doc.theme = theme


def ratings(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    doc.add_root(create_ratings_figure(run_params.game, tags))
    doc.theme = theme

def evaluation(doc):
    tag_str = doc.session_context.request.arguments.get('tags')[0].decode()
    tags = [t for t in tag_str.split(',') if t]
    if not tags:
        return

    benchmark_dict = get_benchmark_eval_mapping(tags)
    benchmark_tags = list(benchmark_dict.keys())
    if not benchmark_tags:
        return
    current_benchmark_tag = benchmark_tags[0]

    select = Select(title="Select Benchmark Tag", value=current_benchmark_tag, options=benchmark_tags)
    plot_container = create_eval_figure(run_params.game, current_benchmark_tag, benchmark_dict[current_benchmark_tag])

    def update_plot(attr, old, new):
        new_plot = create_eval_figure(run_params.game, select.value, benchmark_dict[select.value])
        layout.children[1] = new_plot

    select.on_change('value', update_plot)

    layout = column(select, plot_container)
    doc.add_root(layout)
    doc.theme = theme


class DocumentCollection:
    def __init__(self, tags: List[str]):
        tag_str = ','.join(tags)

        flask_host = request.headers.get('X-Forwarded-Host', request.host)
        protocol = request.headers.get('X-Forwarded-Proto', 'http')

        # For local runs, flask_host will be "127.0.0.1:{flask_port}"
        #
        # On runpod.io, flask_host will be a proxy address like
        #    x2hjevr3ktavjr-{flask_port}.proxy.runpod.net
        #
        # The below rreplace() replaces the port for either case.
        bokeh_host = rreplace(flask_host, str(flask_port), str(bokeh_port), 1)
        assert flask_host != bokeh_host, flask_host

        def make_doc(name):
            return server_document(f'{protocol}://{bokeh_host}/{name}',
                                   arguments={'tags': tag_str})

        training_heads = [(head, make_doc(f'training_{head}')) for head in all_training_heads]
        training = make_doc('training')
        self_play = make_doc('self_play')
        ratings = make_doc('ratings')
        evaluation = make_doc('evaluation')

        self.tags = tags
        self.training_heads = training_heads
        self.training = training
        self.self_play = self_play
        self.ratings = ratings
        self.evaluation = evaluation

    def get_base_data(self):
        return {
            'training_heads': self.training_heads,
            'training': self.training,
            'self_play': self.self_play,
            'ratings': self.ratings,
            'evaluation': self.evaluation,
            'tags': usable_tags,
            'init_tags': self.tags,
        }

    def get_update_data(self):
        d = {
            'training': self.training,
            'self_play': self.self_play,
            'ratings': self.ratings,
            'evaluation': self.evaluation,
            }
        for h, head in enumerate(all_training_heads):
            d[f'training_{head}'] = self.training_heads[h][1]
        return d


@app.route('/', methods=['GET'])
def bkapp_page():
    # Get the tags from the session
    title = f'{run_params.game} Dashboard'
    tags = session.get('tags', default_tags)
    docs = DocumentCollection(tags)
    data = docs.get_base_data()
    data['title'] = title
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
        '/evaluation': evaluation,
    }

    for head in all_training_heads:
        apps[f'/training_{head}'] = training_head(head)

    # In principle, we should set the allow_list in a much more restrictive way.
    # In practice, this may be running on a cloud server like runpod.io, which presents headaches
    # because of their use of proxies. So, we allow all origins for now.
    allow_list = ['*']

    server = Server(apps, io_loop=IOLoop(),
                    allow_websocket_origin=allow_list,
                    port=bokeh_port)

    server.start()
    server.io_loop.start()


def main():
    Thread(target=bk_worker).start()

    print('*********************************************************')
    print('To view the dashboard, open a web browser and navigate to')
    print(f'http://127.0.0.1:{flask_port}/')
    print('*********************************************************')
    print('')

    app.run(host="0.0.0.0", port=flask_port)


if __name__ == '__main__':
    main()
