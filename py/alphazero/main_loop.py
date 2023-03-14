#!/usr/bin/env python3

import argparse
import os
import pipes
import signal
import subprocess
import sys
import time
from typing import List

from alphazero.manager import AlphaZeroManager
from alphazero.optimization_args import add_optimization_args, OptimizationArgs
from config import Config
from util import subprocess_util
from util.py_util import timed_print


class Args:
    c4_base_dir_root: str
    tag: str
    restart_gen: int
    remote_host: str
    remote_repo_path: str
    remote_c4_base_dir_root: str
    self_play_loop: bool
    train_loop: bool
    promote_loop: bool

    @staticmethod
    def load(args):
        Args.c4_base_dir_root = args.c4_base_dir_root
        Args.tag = args.tag
        Args.restart_gen = args.restart_gen
        Args.remote_host = args.remote_host
        Args.remote_repo_path = args.remote_repo_path
        Args.remote_c4_base_dir_root = args.remote_c4_base_dir_root
        Args.self_play_loop = args.self_play_loop
        Args.train_loop = args.train_loop
        Args.promote_loop = args.promote_loop

        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    parser.add_argument('--restart-gen', type=int, help='gen to resume at')
    cfg.add_parser_argument('c4.base_dir_root', parser, '-d', '--c4-base-dir-root', help='base-dir-root for game/model files')
    cfg.add_parser_argument('remote.host', parser, '-H', '--remote-host', help='remote host for model training')
    cfg.add_parser_argument('remote.repo.path', parser, '-P', '--remote-repo-path',
                            help='remote repo path for model training')
    cfg.add_parser_argument('remote.c4.base_dir_root', parser, '-D', '--remote-c4-base-dir-root',
                            help='--c4-base-dir-root on remote host')
    parser.add_argument('--self-play-loop', action='store_true', help='run self-play loop')
    parser.add_argument('--train-loop', action='store_true', help='run train loop')
    parser.add_argument('--promote-loop', action='store_true', help='run promote loop')
    add_optimization_args(parser)

    args = parser.parse_args()
    Args.load(args)
    OptimizationArgs.load(args)


def main():
    load_args()
    c4_base_dir = os.path.join(Args.c4_base_dir_root, Args.tag)
    manager = AlphaZeroManager(c4_base_dir)

    if Args.self_play_loop:
        manager.self_play_loop()
    elif Args.train_loop:
        manager.train_loop()
    elif Args.promote_loop:
        manager.promote_loop()
    else:
        manager.rm_kill_file()
        if Args.restart_gen:
            manager.erase_data_after(Args.restart_gen)

        assert Args.remote_host, 'Required option: -H'
        assert Args.remote_repo_path, 'Required option: -P'
        assert Args.remote_c4_base_dir_root, 'Required option: -D'

        procs: List[subprocess.Popen] = []

        def kill_all():
            for proc in procs:
                timed_print('Killing process: %s' % proc.pid)
                proc.kill()

        signal.signal(signal.SIGINT, lambda *args, **kwargs: kill_all())

        self_play_cmd = ' '.join(map(pipes.quote, sys.argv + ['--self-play-loop']))
        train_cmd = ' '.join(map(pipes.quote, sys.argv + ['--train-loop']))

        remote_promote_cmd = ' '.join(map(
            pipes.quote, sys.argv + ['--promote-loop', '--c4-base-dir-root', Args.remote_c4_base_dir_root]))
        promote_cmd = 'ssh %s "cd %s; %s"' % (Args.remote_host, Args.remote_repo_path, remote_promote_cmd)

        timed_print(f'Running: {self_play_cmd}')
        timed_print(f'Running: {train_cmd}')
        timed_print(f'Running: {promote_cmd}')

        self_play_proc = subprocess_util.Popen(self_play_cmd)
        train_proc = subprocess_util.Popen(train_cmd, stdout=None)  # has the most interesting output
        promote_proc = subprocess_util.Popen(promote_cmd)
        procs = [self_play_proc, train_proc, promote_proc]

        while True:
            if self_play_proc.poll() is not None:
                timed_print('self_play_proc exited')
                break
            if train_proc.poll() is not None:
                timed_print('train_proc exited')
                break
            if promote_proc.poll() is not None:
                timed_print('promote_proc exited')
                break
            time.sleep(1)

        proc_cmd_stdout_list = [
            (self_play_proc, self_play_cmd, manager.get_self_play_stdout()),
            (promote_proc, promote_cmd, manager.get_promote_stdout()),
        ]
        for (proc, cmd, stdout) in proc_cmd_stdout_list:
            if proc.returncode == 0:
                continue
            print('**************************************')
            timed_print('Cmd failed: %s' % cmd)
            print('**************************************')
            tail_cmd = 'tail -n20 %s' % stdout
            print(tail_cmd)
            print('')
            os.system(tail_cmd)

        kill_all()


if __name__ == '__main__':
    main()
