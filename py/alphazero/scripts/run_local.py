#!/usr/bin/env python3

"""
An AlphaZero run has 3 components:

1. loop controller: trains neural net from training data
2. self-play server(s): uses neural net to generate training data
3. [optional] ratings server(s): evaluates neural net against reference agents

These have corresponding launcher scripts in py/alphazero/scripts/:

- run_loop_controller.py
- run_self_play_server.py
- run_ratings_server.py

This script launches 1 server of each type on the local machine, using the above launcher scripts.
Although this is convenient, this setup comes at a cost, particularly if the local machine has fewer
than 3 GPU's. In that case, the different servers need to pause at various points to avoid GPU
contention. The pausing/unpausing logic is all handled automatically by the loop controller, and
can be summarized as follows:

* 1 GPU setup *

Self-play and training run concurrently, sharing the single GPU, with self-play paused whenever
training is running. The ratings server starts paused. Every N generations, the ratings server wakes
up to rate one generation. During this time, both the self-play and training servers are paused.
When the ratings server is done, it goes back to sleep, and the self-play/training servers resume
their concurrent runs. N defaults to 10.

* 2 GPU setup *

Self-play and training run concurrently on separate GPU's, without pausing. The ratings server
starts paused. Every N generations, the ratings server wakes up to rate one generation. During this
time, the self-play server switches over to share the training server's GPU, and they run
concurrently, sharing the single GPU, with self-play paused whenever training is running. When the
ratings server is done, it goes back to sleep, the self-play server moves back to its own GPU, and
the self-play and training servers resume their concurrent runs. N defaults to 10.

* 3+ GPU setup *

Self-play, training, and ratings all run concurrently without interfering with each other.
"""
import argparse
from dataclasses import dataclass
import os
from pipes import quote
import subprocess
import time

import torch

from alphazero.logic.run_params import RunParams
from alphazero.logic import constants
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

    RunParams.add_args(parser)
    TrainingParams.add_args(parser)
    Params.add_args(parser)
    LoggingParams.add_args(parser)

    return parser.parse_args()


def launch_self_play_server(params_dict, cuda_device: int):
    default_self_play_server_params = SelfPlayServerParams()

    params = params_dict['Params']
    logging_params = params_dict['LoggingParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_self_play_server.py',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])
    if default_self_play_server_params.binary_path != params.binary_path:
        cmd.extend(['--binary-path', params.binary_path])

    logging_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching self-play server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int):
    default_loop_controller_params = LoopControllerParams()

    params = params_dict['Params']
    run_params = params_dict['RunParams']
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

    logging_params.add_to_cmd(cmd)
    run_params.add_to_cmd(cmd)
    training_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching loop controller: {cmd}')
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


def main():
    args = load_args()
    run_params = RunParams.create(args)
    training_params = TrainingParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)

    params_dict = {
        'RunParams': run_params,
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
        procs.append(('Self-play', launch_self_play_server(params_dict, min(1, n-1))))

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
