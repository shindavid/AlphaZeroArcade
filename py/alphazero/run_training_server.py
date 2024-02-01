#!/usr/bin/env python3

import argparse

from alphazero.cmd_server import CmdServer
from alphazero.common_args import CommonArgs
from alphazero.directory_organizer import DirectoryOrganizer
from alphazero.training_params import TrainingParams
from alphazero.training_server import TrainingServer
from util.logging_util import configure_logger, get_logger

import os


logger = get_logger()


class Args:
    cmd_server_host: str
    cmd_server_port: int
    cuda_device_str: str
    model_cfg: str
    debug: bool

    @staticmethod
    def load(args):
        Args.cmd_server_host = args.cmd_server_host
        Args.cmd_server_port = args.cmd_server_port
        Args.cuda_device_str = args.cuda_device_str
        Args.model_cfg = args.model_cfg
        Args.debug = bool(args.debug)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--cmd-server-host', type=str, default='localhost',
                            help='cmd-server host (default: %(default)s)')
        parser.add_argument('--cmd-server-port', type=int, default=CmdServer.DEFAULT_PORT,
                            help='cmd-server port (default: %(default)s)')
        parser.add_argument('--cuda-device-str',
                            default='cuda:0', help='cuda device str')
        parser.add_argument('-m', '--model-cfg', default='default',
                            help='model config (default: %(default)s)')
        parser.add_argument('--debug', action='store_true', help='debug mode')


def load_args():
    parser = argparse.ArgumentParser()

    CommonArgs.add_args(parser)
    TrainingParams.add_args(parser)
    Args.add_args(parser)

    args = parser.parse_args()

    CommonArgs.load(args)
    TrainingParams.load(args)
    Args.load(args)


def main():
    load_args()
    log_filename = os.path.join(DirectoryOrganizer().logs_dir, 'training-server.log')
    configure_logger(log_filename, debug=Args.debug)
    logger.info(f'**** Starting training-server ****')
    training_server = TrainingServer(
        Args.cmd_server_host, Args.cmd_server_port, Args.cuda_device_str, Args.model_cfg)
    training_server.run()


if __name__ == '__main__':
    main()
