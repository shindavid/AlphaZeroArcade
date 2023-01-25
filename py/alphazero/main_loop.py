#!/usr/bin/env python3

import argparse

from alphazero import shared
from config import Config


class Args:
    c4_base_dir: str
    remote_host: str
    remote_repo_path: str
    remote_c4_base_dir: str

    @staticmethod
    def load(args):
        Args.c4_base_dir = args.c4_base_dir
        Args.remote_host = args.remote_host
        Args.remote_repo_path = args.remote_repo_path
        Args.remote_c4_base_dir = args.remote_c4_base_dir

        assert Args.c4_base_dir, 'Required option: -d'
        assert Args.remote_host, 'Required option: -H'
        assert Args.remote_repo_path, 'Required option: -P'
        assert Args.remote_c4_base_dir, 'Required option: -D'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    cfg.add_parser_argument('c4.base_dir', parser, '-d', '--c4-base-dir', help='base-dir for game/model files')
    cfg.add_parser_argument('remote.host', parser, '-H', '--remote-host', help='remote host for model training')
    cfg.add_parser_argument('remote.repo.path', parser, '-P', '--remote-path',
                            help='remote repo path for model training')
    cfg.add_parser_argument('remote.c4.base_dir', parser, '-D', '--remote-c4-base-dir',
                            help='--c4-base-dir on remote host')

    return parser.parse_args()


def main():
    load_args()
    manager = shared.AlphaZeroManager(Args.c4_base_dir)
    manager.main_loop(Args.remote_host, Args.remote_repo_path, Args.remote_c4_base_dir)


if __name__ == '__main__':
    main()
