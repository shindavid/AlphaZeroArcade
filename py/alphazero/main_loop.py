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
from dataclasses import dataclass
import os
from pipes import quote
import signal
import subprocess
import time

import torch

from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.logic.cmd_server import CmdServer
from alphazero.logic.sample_window_logic import SamplingParams
from alphazero.logic.learning_params import LearningParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.repo_util import Repo
from util import subprocess_util


logger = get_logger()


@dataclass
class Params:
    cmd_server_port: int = constants.DEFAULT_CMD_SERVER_PORT
    binary_path: str = None
    model_cfg: str = 'default'

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            cmd_server_port=args.cmd_server_port,
            binary_path=args.binary_path,
            model_cfg=args.model_cfg,
            )

    @staticmethod
    def add_args(parser):
        defaults = Params()

        parser.add_argument('--cmd-server-port', type=int, default=defaults.cmd_server_port,
                            help='cmd server port (default: %(default)s)')
        parser.add_argument('-b', '--binary-path',
                            help='binary path. By default, if a unique binary is found in the '
                            'alphazero dir, it will be used. If no binary is found in the alphazero '
                            'dir, then will use one found in REPO_ROOT/target/Release/bin/. If '
                            'multiple binaries are found in the alphazero dir, then this option is '
                            'required.')
        parser.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                            help='model config (default: %(default)s)')


def load_args():
    parser = argparse.ArgumentParser()

    CommonParams.add_args(parser)
    SamplingParams.add_args(parser)
    LearningParams.add_args(parser)
    Params.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def launch_cmd_server(params_dict):
    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    sampling_params = params_dict['SamplingParams']
    logging_params = params_dict['LoggingParams']

    cmd = [
        'py/alphazero/run_cmd_server.py',
        '--port', str(params.cmd_server_port),
        ]
    common_params.add_to_cmd(cmd)
    sampling_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching cmd server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_self_play_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    logging_params = params_dict['LoggingParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/run_self_play_server.py',
        '--cmd-server-port', str(params.cmd_server_port),
        '--cuda-device', cuda_device,
    ]
    if params.binary_path:
        cmd.extend(['--binary-path', params.binary_path])

    logging_params.add_to_cmd(cmd)
    common_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching self play server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_training_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    learning_params = params_dict['LearningParams']
    logging_params = params_dict['LoggingParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/run_training_server.py',
        '--cmd-server-port', str(params.cmd_server_port),
        '--cuda-device', cuda_device,
        '--model-cfg', params.model_cfg,
    ]

    logging_params.add_to_cmd(cmd)
    common_params.add_to_cmd(cmd)
    learning_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching training server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    sampling_params = SamplingParams.create(args)
    learning_params = LearningParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)

    params_dict = {
        'CommonParams': common_params,
        'SamplingParams': sampling_params,
        'LearningParams': learning_params,
        'Params': params,
        'LoggingParams': logging_params,
        }

    configure_logger(params=logging_params)

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

    procs.append(('Cmd-server', launch_cmd_server(params_dict)))
    time.sleep(0.5)  # Give cmd-server time to initialize socket (TODO: fix this hack)
    procs.append(('Self-play', launch_self_play_server(params_dict, n-1)))
    procs.append(('Training', launch_training_server(params_dict, 0)))

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
