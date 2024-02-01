#!/usr/bin/env python3

"""
This script serves as a thin wrapper around a c++ binary. It communicates with the cmd server, and
upon receiving "start" requests from the cmd server, will start the c++ binary. From there, the
c++ binary and the cmd server communicate directly via TCP.

This setup allows us to have multiple c++ binary processes under the hood of a single self play
server process. This is useful because generation 0 uses a different configuration than the other
generations - it was easiest to accomplish this by simply restarting the c++ binary process
between generations 0 and 1. Demanding a restart of the entire self-play server process between
generations 0 and 1 would have been more complicated operationally.
"""
import argparse

from alphazero.cmd_server import CmdServer
from alphazero.common_args import CommonArgs
from alphazero.directory_organizer import DirectoryOrganizer
from alphazero.self_play_server import SelfPlayServer
from util.logging_util import configure_logger, get_logger

import os


logger = get_logger()


class Args:
    """
    TODO: games_base_dir is needed because currently the c++ binary directly writes to the
    filesystem - this assumes that the c++ binary and the training server share the same filesystem.
    Eventually we should have the game data communicated via TCP, with the filesystem write
    happening elsewhere.
    """
    cmd_server_host: str
    cmd_server_port: int
    binary_path: str
    cuda_device: str
    debug: bool

    @staticmethod
    def load(args):
        Args.cmd_server_host = args.cmd_server_host
        Args.cmd_server_port = args.cmd_server_port
        Args.binary_path = args.binary_path
        Args.cuda_device = args.cuda_device
        Args.debug = bool(args.debug)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--cmd-server-host', type=str, default='localhost',
                            help='cmd-server host (default: %(default)s)')
        parser.add_argument('--cmd-server-port', type=int, default=CmdServer.DEFAULT_PORT,
                            help='cmd-server port (default: %(default)s)')

        parser.add_argument('-b', '--binary-path',
                            help='binary path. By default, if a unique binary is found in the '
                            'alphazero dir, it will be used. If no binary is found in the alphazero '
                            'dir, then will use one found in REPO_ROOT/target/Release/bin/. If '
                            'multiple binaries are found in the alphazero dir, then this option is '
                            'required.')
        parser.add_argument('--cuda-device', default='cuda:0',
                            help='cuda device (default: %(default)s)')
        parser.add_argument('--debug', action='store_true', help='debug mode')


def load_args():
    parser = argparse.ArgumentParser()

    CommonArgs.add_args(parser)
    Args.add_args(parser)

    args = parser.parse_args()
    CommonArgs.load(args)
    Args.load(args)


def main():
    load_args()
    log_filename = os.path.join(DirectoryOrganizer().logs_dir, 'self-play-server.log')
    configure_logger(log_filename, debug=Args.debug)
    logger.info(f'**** Starting self-play-server ****')
    server = SelfPlayServer(Args.cmd_server_host, Args.cmd_server_port, Args.cuda_device,
                            Args.binary_path)
    server.run()


if __name__ == '__main__':
    main()
