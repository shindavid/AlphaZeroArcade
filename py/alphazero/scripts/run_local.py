#!/usr/bin/env python3

"""
An AlphaZero run has 2 components:

1. self-play server(s): generates training data
2. loop controller: trains the neural net from the training data

There is only one loop controller per run, while there can be multiple self-play servers.
All the servers can run on the same machine, or on different machines - communication between them
is done via TCP.

This script launches 2 servers on the local machine: one loop controller and one self-play server.
This is useful for dev/testing purposes. By default, the script detects if the local machine has
multiple GPU's or not. If there is a single GPU, then the system is configured to pause the
self-play server whenever a train loop is active.

For standard production runs, you may want multiple self-play servers, on different machines.
"""
import argparse
from dataclasses import dataclass
import os
from pipes import quote
import subprocess
import time

import torch

from alphazero.logic.common_params import CommonParams
from alphazero.logic import constants
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.training_params import TrainingParams
from alphazero.servers.self_play.self_play_server import SelfPlayServerParams
from alphazero.servers.loop_control.loop_controller import LoopControllerParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.repo_util import Repo
from util import subprocess_util


logger = get_logger()


@dataclass
class Params:
    port: int = constants.DEFAULT_LOOP_CONTROLLER_PORT
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
                            help='LoopController port (default: %(default)s)')
        parser.add_argument('-m', '--model-cfg', default=defaults.model_cfg,
                            help='model config (default: %(default)s)')
        parser.add_argument('-b', '--binary-path',
                            help='binary path. Default: last-used binary for this tag. If this is '
                            'the first run for this tag, then target/Release/bin/{game}')


def load_args():
    parser = argparse.ArgumentParser()

    CommonParams.add_args(parser)
    TrainingParams.add_args(parser)
    Params.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def launch_self_play_server(params_dict, cuda_device: int):
    default_self_play_server_params = SelfPlayServerParams()

    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    logging_params = params_dict['LoggingParams']
    organizer = DirectoryOrganizer(common_params)

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_self_play_server.py',
        '--cuda-device', cuda_device,
        '--log-dir', organizer.logs_dir,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])

    common_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching self-play server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int):
    default_loop_controller_params = LoopControllerParams()

    params = params_dict['Params']
    common_params = params_dict['CommonParams']
    training_params = params_dict['TrainingParams']
    logging_params = params_dict['LoggingParams']

    cmd = [
        'py/alphazero/scripts/run_loop_controller.py',
        '--cuda-device', f'cuda:{cuda_device}',
        ]
    if default_loop_controller_params.port != params.port:
        cmd.extend(['--port', str(params.port)])
    if default_loop_controller_params.model_cfg != params.model_cfg:
        cmd.extend(['--model-cfg', params.model_cfg])
    if default_loop_controller_params.binary_path != params.binary_path:
        cmd.extend(['--binary-path', params.binary_path])

    logging_params.add_to_cmd(cmd)
    common_params.add_to_cmd(cmd)
    training_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching loop controller: {cmd}')
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


def main():
    args = load_args()
    common_params = CommonParams.create(args)
    training_params = TrainingParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)

    params_dict = {
        'CommonParams': common_params,
        'TrainingParams': training_params,
        'Params': params,
        'LoggingParams': logging_params,
        }

    configure_logger(params=logging_params, prefix='[main]')

    os.chdir(Repo.root())

    n = torch.cuda.device_count()
    assert n > 0, 'No GPU found'

    procs = []
    try:
        procs.append(('Loop-controller', launch_loop_controller(params_dict, 0)))
        time.sleep(0.5)  # Give loop-controller time to initialize socket (TODO: fix this hack)
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
    finally:
        for descr, proc in procs:
            if proc.poll() is None:
                proc.terminate()
                logger.info(f'Terminated {descr} process {proc.pid}')


if __name__ == '__main__':
    main()
