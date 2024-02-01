#!/usr/bin/env python3

import argparse
import os

from alphazero.logic.common_args import CommonArgs
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.cmd_server import CmdServer
from alphazero.logic.sample_window_logic import SamplingParams
from util.logging_util import configure_logger, get_logger


logger = get_logger()


class Args:
    port: int
    debug: bool

    @staticmethod
    def load(args):
        Args.port = args.port
        Args.debug = bool(args.debug)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--port', type=int, default=CmdServer.DEFAULT_PORT,
                            help='port (default: %(default)s)')
        parser.add_argument('--debug', action='store_true', help='debug mode')


def load_args():
    parser = argparse.ArgumentParser()

    CommonArgs.add_args(parser)
    SamplingParams.add_args(parser)
    Args.add_args(parser)

    args = parser.parse_args()

    CommonArgs.load(args)
    SamplingParams.load(args)
    Args.load(args)


def main():
    load_args()
    log_filename = os.path.join(DirectoryOrganizer().logs_dir, 'cmd-server.log')
    configure_logger(log_filename, debug=Args.debug)

    logger.info(f'**** Starting cmd-server ****')

    cmd_server = CmdServer(Args.port)
    cmd_server.register_signal_handler()
    cmd_server.run()


if __name__ == '__main__':
    main()
