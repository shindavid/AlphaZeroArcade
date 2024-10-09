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
from alphazero.logic.build_params import BuildParams
from alphazero.logic.run_params import RunParams
from alphazero.servers.gaming.ratings_server import RatingsServerParams
from alphazero.servers.gaming.self_play_server import SelfPlayServerParams
from alphazero.servers.loop_control.params import LoopControllerParams
from games.game_spec import GameSpec
import games.index as game_index
from shared.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger, get_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo
from util import subprocess_util

import argparse
from dataclasses import dataclass, fields
import os
from pipes import quote
import subprocess
import time
from typing import Optional

import torch


logger = get_logger()


default_build_params = BuildParams()
default_loop_controller_params = LoopControllerParams()
default_self_play_server_params = SelfPlayServerParams()
default_ratings_server_params = RatingsServerParams()


@dataclass
class Params:
    port: int = default_loop_controller_params.port
    model_cfg: str = default_loop_controller_params.model_cfg
    target_rating_rate: float = default_loop_controller_params.target_rating_rate
    max_positions_per_generation: Optional[int] = \
        default_loop_controller_params.max_positions_per_generation
    rating_tag: str = ''

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        return Params(**kwargs)

    @staticmethod
    def add_args(parser):
        LoopControllerParams.add_args(parser, include_cuda_device=False)
        RatingsServerParams.add_args(parser, omit_base=True)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_training_params = None if game_spec is None else game_spec.training_params
    TrainingParams.add_args(parser, defaults=default_training_params)
    Params.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser)

    return parser.parse_args(), game_spec


def launch_self_play_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_self_play_server.py',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])

    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, add_ffi_lib_path_option=False)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching self-play server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_ratings_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_ratings_server.py',
        '--cuda-device', cuda_device,
    ]
    if default_ratings_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])
    if default_ratings_server_params.rating_tag != params.rating_tag:
        cmd.extend(['--rating-tag', params.rating_tag])

    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, add_ffi_lib_path_option=False)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching ratings server: {cmd}')
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int):
    params = params_dict['Params']
    run_params = params_dict['RunParams']
    game_spec = game_index.get_game_spec(run_params.game)
    default_training_params = game_spec.training_params
    training_params = params_dict['TrainingParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

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
    if default_loop_controller_params.max_positions_per_generation != params.max_positions_per_generation:
        cmd.extend(['--max-positions-per-generation', str(params.max_positions_per_generation)])

    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, add_binary_path_option=False)
    run_params.add_to_cmd(cmd)
    training_params.add_to_cmd(cmd, default_training_params)

    cmd = ' '.join(map(quote, cmd))
    logger.info(f'Launching loop controller: {cmd}')
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


def main():
    args, game_spec = load_args()
    run_params = RunParams.create(args)
    training_params = TrainingParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    params_dict = {
        'RunParams': run_params,
        'TrainingParams': training_params,
        'Params': params,
        'LoggingParams': logging_params,
        'BuildParams': build_params,
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

        if game_spec.reference_player_family is not None:
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
