#!/usr/bin/env python3

"""
An AlphaZero run has 2 components:

1. self-play server(s): generates training data
2. training server: trains the neural net from the training data

There is only one training server per run, while there can be multiple self-play servers.
All the servers can run on the same machine, or on different machines - communication between them
is done via TCP.

This script launches 2 servers on the local machine: one training server and one self-play server.
This is useful for dev/testing purposes. By default, the script detects if the local machine has
multiple GPU's or not. If there is a single GPU, then the system is configured to pause the
self-play server whenever a train loop is active.

For standard production runs, you may want multiple self-play servers, on different machines.
"""
import argparse
from dataclasses import dataclass
import os
from pipes import quote
import signal
import subprocess
import time
from typing import List

import torch

from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.logic.learning_params import LearningParams
from alphazero.logic.sample_window_logic import SamplingParams
from alphazero.logic.self_play_server import SelfPlayServerParams
from alphazero.logic.training_server import TrainingServerParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.repo_util import Repo
from util import subprocess_util


logger = get_logger()


@dataclass
class Params:
    port: int = constants.DEFAULT_TRAINING_SERVER_PORT
    binary_path: str = None
    model_cfg: str = 'default'

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            port=args.port,
            binary_path=args.binary_path,
            model_cfg=args.model_cfg,
            )

    @staticmethod
    def add_args(parser):
        defaults = Params()

        parser.add_argument('--port', type=int,
                            default=defaults.port,
                            help='TrainingServer port (default: %(default)s)')
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


def launch_self_play_server(params_dict, cuda_device: int):
    default_self_play_server_params = SelfPlayServerParams()

    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    logging_params = params_dict['LoggingParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/run_self_play_server.py',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.training_server_port != params.port:
        cmd.extend(['--training_server_port', str(params.port)])
    if default_self_play_server_params.binary_path != params.binary_path:
        cmd.extend(['--binary-path', params.binary_path])

    common_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_training_server(params_dict, cuda_device: int):
    default_training_server_params = TrainingServerParams()

    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    learning_params = params_dict['LearningParams']
    sampling_params = params_dict['SamplingParams']
    logging_params = params_dict['LoggingParams']

    cmd = [
        'py/alphazero/run_training_server.py',
        '--cuda-device', f'cuda:{cuda_device}',
        ]
    if default_training_server_params.port != params.port:
        cmd.extend(['--port', str(params.port)])
    if default_training_server_params.model_cfg != params.model_cfg:
        cmd.extend(['--model-cfg', params.model_cfg])

    logging_params.add_to_cmd(cmd)
    common_params.add_to_cmd(cmd)
    learning_params.add_to_cmd(cmd)
    sampling_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching training server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


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

    configure_logger(params=logging_params, prefix='[main]')

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

    procs.append(('Training', launch_training_server(params_dict, 0)))
    time.sleep(0.5)  # Give training-server time to initialize socket (TODO: fix this hack)
    procs.append(('Self-play', launch_self_play_server(params_dict, n-1)))

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
                if proc.stderr is not None:
                    print(proc.stderr.read())
            else:
                print('*' * 80)
                logger.error(f'{descr} process {proc.pid} exited with code {proc.returncode}')
        time.sleep(1)

    shutdown()


if __name__ == '__main__':
    main()
