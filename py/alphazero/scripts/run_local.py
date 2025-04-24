#!/usr/bin/env python3
"""
An AlphaZero run consists of three primary components:

1. Loop Controller
   - Central coordinator that manages training, task orchestration, and system state.
   - Responsibilities include:
     - Training the neural network from self-play data.
     - Managing self-play, evaluation, and benchmarking processes.
     - Producing benchmark committees.
     - Evaluating models against benchmark committees.
     - Rating models against reference players.

2. Self-Play Server(s)
   - Generates training data by playing games using the current neural network.
   - Can be run in parallel across multiple GPUs for scalability.

3. Rating Servers
   These handle different aspects of model evaluation and benchmarking:
   - Benchmark Server: Evaluates models against themselves to generate a benchmark committee.
   - Eval Server: Evaluates experimental or new runs against the benchmark.
   - Ratings Server: Rates models by playing them against reference players.

Each component has a corresponding launcher script in py/alphazero/scripts/:
- run_loop_controller.py
- run_self_play_server.py
- run_ratings_server.py
- run_benchmark_server.py
- run_eval_server.py

Standard Usage Recipes:

1. Start a new training run:
`./py/alphazero/scripts/run_local.py -g {game} -t {tag}`
    - If there is no default benchmark, it will create a benchmark committee from itself.
    - Otherwise, it will use the default benchmark to evaluate the new run.

2. Set default benchmark:
`./py/alphazero/scripts/set_default_benchmark.py -g {game} -t {tag}`
"""

from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.logic.run_params import RunParams
from alphazero.logic.signaling import register_signal_exception
from alphazero.servers.gaming.ratings_server import RatingsServerParams
from alphazero.servers.gaming.self_play_server import SelfPlayServerParams
from alphazero.servers.loop_control.params import LoopControllerParams
from games.game_spec import GameSpec
import games.index as game_index
from shared.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo
from util import subprocess_util

import argparse
from dataclasses import dataclass, fields
import json
import logging
import os
from pipes import quote
import signal
import subprocess
import time
from typing import Optional

import torch


logger = logging.getLogger(__name__)


default_build_params = BuildParams()
default_loop_controller_params = LoopControllerParams()
default_self_play_server_params = SelfPlayServerParams()
default_ratings_server_params = RatingsServerParams()


@dataclass
class Params:
    port: int = default_loop_controller_params.port
    model_cfg: str = default_loop_controller_params.model_cfg
    target_rating_rate: float = default_loop_controller_params.target_rating_rate
    rating_tag: str = ''
    num_cuda_devices_to_use: Optional[int] = None

    benchmark_tag: Optional[str] = default_loop_controller_params.benchmark_tag

    run_ratings_server: bool = False
    skip_self_play: bool = False

    @staticmethod
    def create(args) -> 'Params':
        kwargs = {f.name: getattr(args, f.name) for f in fields(Params)}
        return Params(**kwargs)

    @staticmethod
    def add_args(parser):
        LoopControllerParams.add_args(parser, include_cuda_device=False)
        RatingsServerParams.add_args(parser, omit_base=True)

        group = parser.add_argument_group('run_local.py options')

        defaults = Params()
        group.add_argument('-C', '--num-cuda-devices-to-use', type=int,
                           default=defaults.num_cuda_devices_to_use,
                           help='Num cuda devices to use (default: all)')
        group.add_argument('--run-ratings-server', action='store_true',
                            help='Run the ratings server')
        group.add_argument('--skip-self-play', action='store_true',
                            help='Do not run the self-play server')


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_training_params = None if game_spec is None else game_spec.training_params
    TrainingParams.add_args(parser, defaults=default_training_params)
    Params.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, loop_controller=True)

    return parser.parse_args(), game_spec


def launch_self_play_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    docker_params = params_dict['DockerParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_self_play_server.py',
        '--ignore-sigint',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching self-play server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_ratings_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    docker_params = params_dict['DockerParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_ratings_server.py',
        '--ignore-sigint',
        '--cuda-device', cuda_device,
    ]
    if default_ratings_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])
    if default_ratings_server_params.rating_tag != params.rating_tag:
        cmd.extend(['--rating-tag', params.rating_tag])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching ratings server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_benchmark_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    docker_params = params_dict['DockerParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_benchmark_server.py',
        '--ignore-sigint',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching benchmark server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_eval_server(params_dict, cuda_device: int):
    params = params_dict['Params']
    docker_params = params_dict['DockerParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cuda_device = f'cuda:{cuda_device}'

    cmd = [
        'py/alphazero/scripts/run_eval_server.py',
        '--ignore-sigint',
        '--cuda-device', cuda_device,
    ]
    if default_self_play_server_params.loop_controller_port != params.port:
        cmd.extend(['--loop_controller_port', str(params.port)])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching eval server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int):
    params = params_dict['Params']
    run_params = params_dict['RunParams']
    game_spec = game_index.get_game_spec(run_params.game)
    default_training_params = game_spec.training_params
    training_params = params_dict['TrainingParams']
    docker_params = params_dict['DockerParams']
    logging_params = params_dict['LoggingParams']
    build_params = params_dict['BuildParams']

    cmd = [
        'py/alphazero/scripts/run_loop_controller.py',
        '--ignore-sigint',
        '--cuda-device', f'cuda:{cuda_device}',
        ]
    if default_loop_controller_params.port != params.port:
        cmd.extend(['--port', str(params.port)])
    if default_loop_controller_params.model_cfg != params.model_cfg:
        cmd.extend(['--model-cfg', params.model_cfg])
    if default_loop_controller_params.target_rating_rate != params.target_rating_rate:
        cmd.extend(['--target-rating-rate', str(params.target_rating_rate)])

    benchmark_tag = get_benchmark_tag(run_params, params)
    if benchmark_tag:
        cmd.extend(['--benchmark-tag', benchmark_tag])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, loop_controller=True)
    run_params.add_to_cmd(cmd)
    training_params.add_to_cmd(cmd, default_training_params)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching loop controller: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


def load_benchmark_info(game: str):
    """
    Load the default benchmark tag for a given game from a JSON file.

    This will read the file:
        /workspace/output/{game}/benchmark_info.json
    """

    file_path = os.path.join("/workspace/output", game, "benchmark_info.json")

    if not os.path.exists(file_path):
        print(f"No benchmark info found for game '{game}'.")
        return None

    with open(file_path, 'r') as f:
        benchmark_info = json.load(f)

    return benchmark_info.get("benchmark_tag")


def get_benchmark_tag(run_params: RunParams, params: Params) -> Optional[str]:
    benchmark_tag = params.benchmark_tag
    if benchmark_tag is None:
        benchmark_tag = load_benchmark_info(run_params.game)
    return benchmark_tag


def main():
    args, game_spec = load_args()

    docker_params = DockerParams.create(args)
    run_params = RunParams.create(args)
    training_params = TrainingParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    params_dict = {
        'RunParams': run_params,
        'TrainingParams': training_params,
        'Params': params,
        'DockerParams': docker_params,
        'LoggingParams': logging_params,
        'BuildParams': build_params,
        }

    configure_logger(params=logging_params, prefix='[main]')

    os.chdir(Repo.root())

    organizer = DirectoryOrganizer(run_params, base_dir_root='/workspace')
    if not organizer.version_check():
        print('The following output directory is outdated:\n')
        print(organizer.base_dir + '\n')
        print('As a result, you cannot resume a run from this directory.')
        print('Please try again with a new tag.')
        return

    n = torch.cuda.device_count()
    assert n > 0, 'No GPU found. Try exiting and relaunching run_docker.py'

    if params.num_cuda_devices_to_use is not None:
        n = params.num_cuda_devices_to_use

    loop_controller_gpu = 0
    self_play_gpus = list(range(n))
    ratings_gpu = n - 1

    register_signal_exception(signal.SIGINT, KeyboardInterrupt,
                              echo_action=lambda: logger.info('Ignoring repeat Ctrl-C'))

    benchmark_tag = get_benchmark_tag(run_params, params)

    procs = []
    try:
        procs.append(('Loop-controller', launch_loop_controller(params_dict, loop_controller_gpu)))
        time.sleep(0.5)  # Give loop-controller time to initialize socket (TODO: fix this hack)
        if not params.skip_self_play:
            for self_play_gpu in self_play_gpus:
                procs.append(('Self-play', launch_self_play_server(params_dict, self_play_gpu)))

        if benchmark_tag is None:
            procs.append(('Benchmark', launch_benchmark_server(params_dict, ratings_gpu)))
        else:
            procs.append(('Eval', launch_eval_server(params_dict, ratings_gpu)))

        if params.run_ratings_server and game_spec.reference_player_family is not None:
            procs.append(('Ratings', launch_ratings_server(params_dict, ratings_gpu)))

        loop = True
        while loop:
            for descr, proc in procs:
                if proc.poll() is None:
                    continue
                loop = False
                if proc.returncode != 0:
                    print('*' * 80)
                    logger.error('%s process %s exited with code %s', descr, proc.pid,
                                 proc.returncode)
                    print('*' * 80)
                    if proc.stderr is not None:
                        print(proc.stderr.read())
                else:
                    print('*' * 80)
                    logger.error('%s process %s exited with code %s', descr, proc.pid,
                                 proc.returncode)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Caught Ctrl-C')
    except:
        logger.error('Unexpected error:', exc_info=True)
    finally:
        for descr, proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                    logger.info('Terminated %s process %s', descr, proc.pid)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    logger.warning('Forcibly killed %s process %s due to time-out during terminate',
                                   descr, proc.pid)


if __name__ == '__main__':
    main()

