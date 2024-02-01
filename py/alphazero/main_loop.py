#!/usr/bin/env python3

"""
An AlphaZero run has 3 components:

1. self-play server(s): generates training data
2. training server(s): trains the neural net from the training data
3. cmd server: coordinates the training and self-play servers

There is only one cmd server per run, while there can be multiple training and self-play servers.
All the servers can run on the same machine, or on different machines - communication between them
is done via TCP.

This script launches 3 servers on the local machine: one cmd server, one training server, and one
self-play server. This is useful for dev/testing purposes. By default, the script detects if the
local machine has multiple GPU's or not. If there is a single GPU, then the system is configured to
pause the self-play server whenever the training server is running.

For standard production runs, you likely still only need one training server, but you will want
multiple self-play servers, on different machines. Only for very large runs will you want multiple
training servers - the coordination between them is not yet implemented.
"""
import argparse
import os
from pipes import quote
import signal
import subprocess
import time
from typing import List

import torch

from alphazero.cmd_server import CmdServer
from alphazero.common_args import CommonArgs
from alphazero.training_params import TrainingParams
from alphazero.sample_window_logic import SamplingParams
from util.logging_util import configure_logger, get_logger
from util.repo_util import Repo
from util.socket_util import is_port_open
from util import subprocess_util


logger = get_logger()


class Args:
    cmd_server_port: int
    binary_path: str
    model_cfg: str
    debug: bool

    @staticmethod
    def load(args):
        Args.cmd_server_port = args.cmd_server_port
        Args.binary_path = args.binary_path
        Args.model_cfg = args.model_cfg
        Args.debug = bool(args.debug)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--cmd-server-port', type=int, default=CmdServer.DEFAULT_PORT,
                            help='cmd server port (default: %(default)s)')
        parser.add_argument('-b', '--binary-path',
                            help='binary path. By default, if a unique binary is found in the '
                            'alphazero dir, it will be used. If no binary is found in the alphazero '
                            'dir, then will use one found in REPO_ROOT/target/Release/bin/. If '
                            'multiple binaries are found in the alphazero dir, then this option is '
                            'required.')
        parser.add_argument('-m', '--model-cfg', default='default',
                            help='model config (default: %(default)s)')
        parser.add_argument('--debug', action='store_true', help='debug mode')

    # @staticmethod
    # def add_to_cmd(cmd: List[str]):
    #     cmd.extend(['--cmd-server-port', str(Args.cmd_server_port)])
    #     if Args.binary_path:
    #         cmd.extend(['--binary-path', Args.binary_path])
    #     cmd.extend(['--model-cfg', Args.model_cfg])
    #     if Args.debug:
    #         cmd.append('--debug')


def load_args():
    parser = argparse.ArgumentParser()

    CommonArgs.add_args(parser)
    SamplingParams.add_args(parser)
    TrainingParams.add_args(parser)
    Args.add_args(parser)

    args = parser.parse_args()

    CommonArgs.load(args)
    SamplingParams.load(args)
    TrainingParams.load(args)
    Args.load(args)


def launch_cmd_server():
    cmd = [
        'py/alphazero/run_cmd_server.py',
        '--port', str(Args.cmd_server_port),
        ]
    if Args.debug:
        cmd.append('--debug')
    CommonArgs.add_to_cmd(cmd)
    SamplingParams.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching cmd server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_self_play_server(cuda_device: int):
    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/run_self_play_server.py',
        '--cmd-server-port', str(Args.cmd_server_port),
        '--cuda-device', cuda_device,
    ]
    if Args.debug:
        cmd.append('--debug')
    if Args.binary_path:
        cmd.extend(['--binary-path', Args.binary_path])
    CommonArgs.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching self play server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_training_server(cuda_device: int):
    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/run_training_server.py',
        '--cmd-server-port', str(Args.cmd_server_port),
        '--cuda-device', cuda_device,
        '--model-cfg', Args.model_cfg,
    ]
    if Args.debug:
        cmd.append('--debug')
    CommonArgs.add_to_cmd(cmd)
    TrainingParams.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching training server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    load_args()
    configure_logger(debug=Args.debug)

    os.chdir(Repo.root())

    n = torch.cuda.device_count()
    assert n > 0, 'No GPU found'

    procs = []

    def shutdown():
        for descr, proc in procs:
            if proc.poll() is None:
                proc.terminate()
                logger.info(f'Terminated {descr} process {proc.pid}')

    def signal_handler(sig, frame):
        logger.info(f'Received signal {sig}')
        shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    procs.append(('Cmd-server', launch_cmd_server()))
    time.sleep(0.5)  # Give cmd-server time to initialize socket
    procs.append(('Self-play', launch_self_play_server(n-1)))
    procs.append(('Training', launch_training_server(0)))

    loop = True
    while loop:
        for descr, proc in procs:
            if proc.poll() is None:
                continue
            loop = False
            if proc.returncode != 0:
                print('*' * 80)
                logger.error(f'{descr} process {proc.pid} exited with code {proc.returncode}')
                print('*' * 80)
                print(proc.stderr.read())
            else:
                print('*' * 80)
                logger.error(f'{descr} process {proc.pid} exited with code {proc.returncode}')
        time.sleep(1)

    shutdown()



if __name__ == '__main__':
    main()
