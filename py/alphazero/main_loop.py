#!/usr/bin/env python3

import argparse
import pipes
import signal
import subprocess
import sys
from typing import List

from alphazero.manager import AlphaZeroManager
from alphazero.optimization_args import add_optimization_args, OptimizationArgs
from config import Config
from util import subprocess_util
from util.py_util import timed_print


class Args:
    c4_base_dir: str
    remote_host: str
    remote_repo_path: str
    remote_c4_base_dir: str
    self_play_loop: bool
    train_loop: bool
    promote_loop: bool

    @staticmethod
    def load(args):
        Args.c4_base_dir = args.c4_base_dir
        Args.remote_host = args.remote_host
        Args.remote_repo_path = args.remote_repo_path
        Args.remote_c4_base_dir = args.remote_c4_base_dir
        Args.self_play_loop = args.self_play_loop
        Args.train_loop = args.train_loop
        Args.promote_loop = args.promote_loop

        assert Args.c4_base_dir, 'Required option: -d'
        assert Args.remote_host, 'Required option: -H'
        assert Args.remote_repo_path, 'Required option: -P'
        assert Args.remote_c4_base_dir, 'Required option: -D'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    cfg.add_parser_argument('c4.base_dir', parser, '-d', '--c4-base-dir', help='base-dir for game/model files')
    cfg.add_parser_argument('remote.host', parser, '-H', '--remote-host', help='remote host for model training')
    cfg.add_parser_argument('remote.repo.path', parser, '-P', '--remote-repo-path',
                            help='remote repo path for model training')
    cfg.add_parser_argument('remote.c4.base_dir', parser, '-D', '--remote-c4-base-dir',
                            help='--c4-base-dir on remote host')
    parser.add_argument('--self-play-loop', action='store_true', help='run self-play loop')
    parser.add_argument('--train-loop', action='store_true', help='run train loop')
    parser.add_argument('--promote-loop', action='store_true', help='run promote loop')
    add_optimization_args(parser)

    args = parser.parse_args()
    Args.load(args)
    OptimizationArgs.load(args)


def main():
    load_args()
    manager = AlphaZeroManager(Args.c4_base_dir)

    if Args.self_play_loop:
        manager.self_play_loop()
    elif Args.train_loop:
        manager.train_loop()
    elif Args.promote_loop:
        manager.promote_loop()
    else:
        manager.rm_kill_file()
        procs: List[subprocess.Popen] = []

        def kill_all():
            for proc in procs:
                timed_print('Killing process: %s' % proc.pid)
                proc.kill()

        signal.signal(signal.SIGINT, lambda *args, **kwargs: kill_all())

        self_play_cmd = ' '.join(map(pipes.quote, sys.argv + ['--self-play-loop']))
        train_cmd = ' '.join(map(pipes.quote, sys.argv + ['--train-loop']))
        remote_promote_cmd = ' '.join(map(pipes.quote, sys.argv + ['--promote-loop', '--c4-base-dir', Args.remote_c4_base_dir]))
        promote_cmd = 'ssh %s %s' % (Args.remote_host, remote_promote_cmd)

        timed_print(f'Running: {self_play_cmd}')
        timed_print(f'Running: {train_cmd}')
        timed_print(f'Running: {promote_cmd}')
        procs.append(subprocess_util.Popen(self_play_cmd))
        procs.append(subprocess_util.Popen(train_cmd, stdout=None))  # has the most interesting output
        procs.append(subprocess_util.Popen(promote_cmd))

        for proc in procs:
            proc.wait()


if __name__ == '__main__':
    main()
