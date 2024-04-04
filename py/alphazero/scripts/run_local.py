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
If the local machine does not have enough GPU's to dedicate one to each server, then the servers
share the GPU's, managing contention via GpuContentionManager.
"""
import argparse
from dataclasses import dataclass
import os
from pipes import quote
import subprocess
import time

import torch

from alphazero.logic.run_params import RunParams
from alphazero.logic.training_params import TrainingParams
from alphazero.servers.ratings.ratings_server import RatingsServerParams
from alphazero.servers.self_play.self_play_server import SelfPlayServerParams
from alphazero.servers.loop_control.params import LoopControllerParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.repo_util import Repo
from util import subprocess_util


logger = get_logger()


default_loop_controller_params = LoopControllerParams()
default_self_play_server_params = SelfPlayServerParams()
default_ratings_server_params = RatingsServerParams()


@dataclass
class Params:
    port: int = default_loop_controller_params.port
    model_cfg: str = default_loop_controller_params.model_cfg
    target_rating_rate: float = default_loop_controller_params.target_rating_rate
    n_search_threads: int = default_ratings_server_params.n_search_threads
    parallelism_factor: int = default_ratings_server_params.parallelism_factor
    binary_path: str = None

    @staticmethod
    def create(args) -> 'Params':
        return Params(
            port=args.port,
            model_cfg=args.model_cfg,
            target_rating_rate=args.target_rating_rate,
            n_search_threads=args.n_search_threads,
            parallelism_factor=args.parallelism_factor,
            binary_path=args.binary_path,
        )

    @staticmethod
    def add_args(parser):
        LoopControllerParams.add_args(parser, include_cuda_device=False)
        RatingsServerParams.add_args(parser, omit_base=True)

        group = parser.add_argument_group('SelfPlayServer/RatingsServer options')
        group.add_argument('-b', '--binary-path',
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


def launch_ratings_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    logging_params = params_dict['LoggingParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_ratings_server.py',
        '--cuda-device', cuda_device,
    ]
    if default_ratings_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])
    if default_ratings_server_params.binary_path != params.binary_path:
        cmd.extend(['--binary-path', params.binary_path])
    if default_ratings_server_params.n_search_threads != params.n_search_threads:
        cmd.extend(['--n_search_threads', str(params.n_search_threads)])
    if default_ratings_server_params.parallelism_factor != params.parallelism_factor:
        cmd.extend(['--parallelism_factor', str(params.parallelism_factor)])

    logging_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching ratings server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int):
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
    if default_loop_controller_params.target_rating_rate != params.target_rating_rate:
        cmd.extend(['--target-rating-rate', str(params.target_rating_rate)])

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

    if n == 1:
        loop_controller_gpu = 0
        self_play_gpu = 0
        ratings_gpu = 0
    elif n == 2:
        loop_controller_gpu = 0
        self_play_gpu = 1
        ratings_gpu = 0
    else:
        loop_controller_gpu = 0
        self_play_gpu = 1
        ratings_gpu = 2

    procs = []
    try:
        procs.append(('Loop-controller', launch_loop_controller(params_dict, loop_controller_gpu)))
        time.sleep(0.5)  # Give loop-controller time to initialize socket (TODO: fix this hack)
        procs.append(('Self-play', launch_self_play_server(params_dict, self_play_gpu)))
        procs.append(('Ratings', launch_ratings_server(params_dict, ratings_gpu)))

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
