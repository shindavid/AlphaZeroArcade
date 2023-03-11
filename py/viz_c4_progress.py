"""
Use this script to visualize the progress of a C4 alphazero run.

Usage:

cd py;

./alphazero/main_loop.py -t <TAG>

While the above is running...

bokeh serve --show viz_c4_progress.py --args -t <TAG>
"""
import argparse
import os

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeSlider, CheckboxGroup
from bokeh.plotting import figure, curdoc
from natsort import natsorted

from config import Config


class Args:
    c4_base_dir_root: str
    tag: str

    @staticmethod
    def load(args):
        Args.c4_base_dir_root = args.c4_base_dir_root
        Args.tag = args.tag

        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    cfg.add_parser_argument('c4.base_dir_root', parser, '-d', '--c4-base-dir-root', help='base-dir-root for game/model files')

    args = parser.parse_args()
    Args.load(args)


load_args()
c4_base_dir = os.path.join(Args.c4_base_dir_root, Args.tag)


class ProgressVisualizer:
    def __init__(self, manager_path: str):
        self.manager_path = manager_path
        self.gating_logs_dir = os.path.join(manager_path, 'gating-logs')

        self.max_gen = 0
        self.den = np.zeros((self.max_gen, 42, 22))
        self.baseline = np.zeros((self.max_gen, 42, 22))
        self.net_t0 = np.zeros((self.max_gen, 42, 22))
        self.net_t1 = np.zeros((self.max_gen, 42, 22))
        self.mcts_t0 = np.zeros((self.max_gen, 42, 22))
        self.mcts_t1 = np.zeros((self.max_gen, 42, 22))

        self.source = ColumnDataSource()
        self.data = {}

    def resize(self, gen: int):
        if gen <= self.max_gen:
            return

        prev_max_gen = self.max_gen
        self.max_gen = gen

        den = np.zeros((self.max_gen, 42, 22))
        baseline = np.zeros((self.max_gen, 42, 22))
        net_t0 = np.zeros((self.max_gen, 42, 22))
        net_t1 = np.zeros((self.max_gen, 42, 22))
        mcts_t0 = np.zeros((self.max_gen, 42, 22))
        mcts_t1 = np.zeros((self.max_gen, 42, 22))

        den[:prev_max_gen] = self.den
        baseline[:prev_max_gen] = self.baseline
        net_t0[:prev_max_gen] = self.net_t0
        net_t1[:prev_max_gen] = self.net_t1
        mcts_t0[:prev_max_gen] = self.mcts_t0
        mcts_t1[:prev_max_gen] = self.mcts_t1

        self.den = den
        self.baseline = baseline
        self.net_t0 = net_t0
        self.net_t1 = net_t1
        self.mcts_t0 = mcts_t0
        self.mcts_t1 = mcts_t1

    def parse_gating_log(self, gen: int, log_filename: str):
        valid = False
        with open(log_filename, 'r') as f:
            for line in f:
                if line.startswith('OracleGrader done'):
                    valid = True
                    break

        if not valid:
            # log appears to be mid-write
            return

        self.resize(gen - 1)
        g = gen - 2
        with open(log_filename, 'r') as f:
            for line in f:
                if not line.startswith('OracleGrader '):
                    continue
                """
                OracleGrader 23-5 count:7 net-t0:0.142857 net-t1:0.139192 mcts-t0:0.142857 mcts-t1:0.128450 baseline:0.177551
                """
                tokens = line.split()
                if tokens[1] == 'overall':
                    continue
                if tokens[1] == 'done':
                    return
                move_num, score = map(int, tokens[1].split('-'))
                d = {}
                for token in tokens[2:]:
                    subtokens = token.split(':')
                    d[subtokens[0]] = float(subtokens[1])

                m = move_num - 1
                s = score

                self.den[g, m, s] += d['count']
                self.baseline[g, m, s] += d['baseline'] * d['count']
                self.net_t0[g, m, s] += d['net-t0'] * d['count']
                self.net_t1[g, m, s] += d['net-t1'] * d['count']
                self.mcts_t0[g, m, s] += d['mcts-t0'] * d['count']
                self.mcts_t1[g, m, s] += d['mcts-t1'] * d['count']

    def refresh(self):
        self.den *= 0
        self.baseline *= 0
        self.net_t0 *= 0
        self.net_t1 *= 0
        self.mcts_t0 *= 0
        self.mcts_t1 *= 0

        for filename in natsorted(os.listdir(self.gating_logs_dir)):
            # gen-1-epoch-2.txt
            full_filename = os.path.join(self.gating_logs_dir, filename)
            gen = int(filename.split('-')[1])
            assert gen > 1, full_filename
            self.parse_gating_log(gen, full_filename)

        n = self.den.shape[0]
        x = np.arange(n)
        den = self.den.sum(axis=(1, 2))

        b = self.baseline.sum(axis=(1, 2)) / den
        n0 = self.net_t0.sum(axis=(1, 2)) / den
        n1 = self.net_t1.sum(axis=(1, 2)) / den
        m0 = self.mcts_t0.sum(axis=(1, 2)) / den
        m1 = self.mcts_t1.sum(axis=(1, 2)) / den

        self.data = {
            # 'n': n,
            'x': x,
            'den': den,
            'b': b,
            'n0': n0,
            'n1': n1,
            'm0': m0,
            'm1': m1
        }

    def plot(self):
        self.refresh()

        move_number = RangeSlider(title='Move-number', start=1, end=42, step=1, value=(1, 42))
        moves_to_win = RangeSlider(title='Moves-to-win', start=1, end=21, step=1, value=(1, 21))
        LABELS = ["Winning Positions", "Drawn Positions"]
        checkbox_group = CheckboxGroup(labels=LABELS, active=[0, 1])

        source = self.source
        source.data = dict(self.data)
        x = self.data['x']

        plot = figure(height=600, width=800, title='Mistake Rate', x_range=[x[0], x[-1]], y_range=[0, 1],
                      x_axis_label='Generation', y_axis_label='Mistake Rate')
        plot.line('x', 'b', source=source, line_color='blue', legend_label='baseline')
        plot.line('x', 'n1', source=source, line_color='green', legend_label='net (temp=1)')
        plot.line('x', 'n0', source=source, line_color='green', line_dash='dashed', legend_label='net (temp=0)')
        plot.line('x', 'm1', source=source, line_color='red', legend_label='mcts (temp=1)')
        plot.line('x', 'm0', source=source, line_color='red', line_dash='dashed', legend_label='mcts (temp=0)')
        plot.add_layout(plot.legend[0], 'right')

        def update_data(attr, old, new):
            mn0, mn1 = move_number.value
            mw0, mw1 = moves_to_win.value
            include_wins = 0 in checkbox_group.active
            include_draws = 1 in checkbox_group.active

            moves_to_win.disabled = not include_wins
            moves_to_win.visible = include_wins

            mn0 -= 1
            mw1 += 1

            mn0 = int(round(mn0))
            mn1 = int(round(mn1))
            mw0 = int(round(mw0))
            mw1 = int(round(mw1))

            mw_list = []
            if include_draws:
                mw_list.append(0)
            if include_wins:
                mw_list.extend(range(mw0, mw1))

            den = self.den[:, mn0:mn1, mw_list].sum(axis=(1, 2))
            b = self.baseline[:, mn0:mn1, mw_list].sum(axis=(1, 2)) / den
            n0 = self.net_t0[:, mn0:mn1, mw_list].sum(axis=(1, 2)) / den
            n1 = self.net_t1[:, mn0:mn1, mw_list].sum(axis=(1, 2)) / den
            m0 = self.mcts_t0[:, mn0:mn1, mw_list].sum(axis=(1, 2)) / den
            m1 = self.mcts_t1[:, mn0:mn1, mw_list].sum(axis=(1, 2)) / den

            source.data = dict(x=x, b=b, n0=n0, n1=n1, m0=m0, m1=m1)

        widgets = [move_number, moves_to_win]
        for widget in widgets:
            widget.on_change('value', update_data)
        checkbox_group.on_change('active', update_data)

        inputs = column(plot, checkbox_group, move_number, moves_to_win)
        return inputs


viz = ProgressVisualizer(c4_base_dir)


curdoc().add_root(viz.plot())
