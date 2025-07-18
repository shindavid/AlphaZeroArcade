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
    ./py/alphazero/scripts/run_local.py -g {game} -t {tag}

    Wait for a few generations to be written, and then launch the dashboard in a separate tab to visualize:

    ./py/alphazero/scripts/launch_dashboard.py -g {game}

    The dashboard has an "Evaluation" tab, which shows the skill progression of the run.

    On your first run, you won't have a benchmark established. If this is the case, the
    "Evaluation" tab will show the results when agents are scored against themselves.

2. Promote that run to become a benchmark:

    ./py/alphazero/scripts/benchmark_tag_local.py -g {game} -t {tag}

    Once promoted, future runs will be rated relative to this run.

"""

from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.logic.build_params import BuildParams
from alphazero.logic.docker_utils import DockerParams, validate_docker_image
from alphazero.logic.run_params import RunParams
from alphazero.logic.signaling import register_signal_exception
from alphazero.servers.gaming.ratings_server import RatingsServerParams
from alphazero.servers.gaming.self_play_server import SelfPlayServerParams
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.params import LoopControllerParams
from games.game_spec import GameSpec
import games.index as game_index
from shared.rating_params import RatingParams
from shared.training_params import TrainingParams
from util.logging_util import LoggingParams, configure_logger
from util.py_util import CustomHelpFormatter
from util.repo_util import Repo
from util import subprocess_util

import atexit
import argparse
from dataclasses import dataclass, fields
import logging
import os
from pipes import quote
import signal
import subprocess
import sys
import time
import torch
from typing import Optional


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
    simulate_cloud: bool = default_loop_controller_params.simulate_cloud
    task_mode: bool = default_loop_controller_params.task_mode

    run_ratings_server: bool = False
    run_benchmark_server: bool = False
    run_eval_server: bool = False

    benchmark_until_gen_gap: int = default_loop_controller_params.benchmark_until_gen_gap

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
        group.add_argument('--run-benchmark-server', action='store_true', help=argparse.SUPPRESS)
        group.add_argument('--run-eval-server', action='store_true', help=argparse.SUPPRESS)


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

    game_spec: Optional[GameSpec] = RunParams.add_args(parser)
    default_training_params = None if game_spec is None else game_spec.training_params
    default_rating_params = None if game_spec is None else game_spec.rating_params
    TrainingParams.add_args(parser, defaults=default_training_params)
    Params.add_args(parser)
    DockerParams.add_args(parser)
    LoggingParams.add_args(parser)
    BuildParams.add_args(parser, loop_controller=True)
    RatingParams.add_args(parser, defaults=default_rating_params)

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
    rating_params: RatingParams = params_dict['RatingParams']

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
    rating_params.add_to_cmd(cmd, server=True)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching ratings server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_benchmark_server(params_dict, cuda_device: int, game_spec: GameSpec):
    params: Params = params_dict['Params']
    docker_params: DockerParams = params_dict['DockerParams']
    logging_params: LoggingParams = params_dict['LoggingParams']
    build_params: BuildParams = params_dict['BuildParams']
    rating_params: RatingParams = params_dict['RatingParams']

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
    rating_params.add_to_cmd(cmd, server=True)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching benchmark server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_eval_server(params_dict, cuda_device: int, game_spec: GameSpec):
    params: Params = params_dict['Params']
    docker_params: DockerParams = params_dict['DockerParams']
    logging_params: LoggingParams = params_dict['LoggingParams']
    build_params: BuildParams = params_dict['BuildParams']
    rating_params: RatingParams = params_dict['RatingParams']

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
    rating_params.add_to_cmd(cmd, server=True)

    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching eval server: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def launch_loop_controller(params_dict, cuda_device: int, benchmark_tag: Optional[str],
                           game_spec: GameSpec):
    params: Params = params_dict['Params']
    run_params: RunParams = params_dict['RunParams']
    game_spec = game_index.get_game_spec(run_params.game)
    default_training_params = game_spec.training_params
    training_params: TrainingParams = params_dict['TrainingParams']
    docker_params: DockerParams = params_dict['DockerParams']
    logging_params: LoggingParams = params_dict['LoggingParams']
    build_params: BuildParams = params_dict['BuildParams']
    rating_params: RatingParams = params_dict['RatingParams']

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

    if rating_params.target_elo_gap is not None:
        cmd.extend(['--target-elo-gap', str(rating_params.target_elo_gap)])
    else:
        cmd.extend(['--target-elo-gap', str(rating_params.default_target_elo_gap.first_run)])

    if default_loop_controller_params.benchmark_until_gen_gap != params.benchmark_until_gen_gap:
        cmd.extend(['--benchmark-until-gen-gap', str(params.benchmark_until_gen_gap)])

    if params.task_mode:
        cmd.extend(['--task-mode'])

    if benchmark_tag:
        cmd.extend(['--benchmark-tag', benchmark_tag])

    if params.simulate_cloud:
        cmd.extend(['--simulate-cloud'])

    docker_params.add_to_cmd(cmd)
    logging_params.add_to_cmd(cmd)
    build_params.add_to_cmd(cmd, loop_controller=True)
    run_params.add_to_cmd(cmd)
    training_params.add_to_cmd(cmd, default_training_params)
    rating_params.add_to_cmd(cmd, loop_controller=True)
    cmd = ' '.join(map(quote, cmd))
    logger.info('Launching loop controller: %s', cmd)
    return subprocess_util.Popen(cmd, stdout=None, stderr=None)


def main():
    args, game_spec = load_args()

    docker_params = DockerParams.create(args)
    run_params = RunParams.create(args)
    training_params = TrainingParams.create(args)
    params = Params.create(args)
    logging_params = LoggingParams.create(args)
    build_params = BuildParams.create(args)
    rating_params = RatingParams.create(args)

    if not docker_params.skip_image_version_check:
        validate_docker_image()

    params_dict = {
        'RunParams': run_params,
        'TrainingParams': training_params,
        'Params': params,
        'DockerParams': docker_params,
        'LoggingParams': logging_params,
        'BuildParams': build_params,
        'RatingParams': rating_params,
        }

    configure_logger(params=logging_params, prefix='[run_local]')

    os.chdir(Repo.root())

    organizer = DirectoryOrganizer(run_params, base_dir_root=Workspace)
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

    benchmark_tag = params.benchmark_tag

    descs = []
    procs = []
    try:
        organizer.assert_unlocked()

        descs.append('Loop-controller')
        p = launch_loop_controller(params_dict, loop_controller_gpu, benchmark_tag, game_spec)
        procs.append(p)
        time.sleep(0.5)  # Give loop-controller time to initialize socket (TODO: fix this hack)
        if not params.task_mode:
            for self_play_gpu in self_play_gpus:
                descs.append('Self-play')
                procs.append(launch_self_play_server(params_dict, self_play_gpu))

        if params.task_mode:
            if params.run_benchmark_server:
                descs.append('Benchmark')
                procs.append(launch_benchmark_server(params_dict, ratings_gpu, game_spec))
            if params.run_eval_server:
                descs.append('Eval')
                procs.append(launch_eval_server(params_dict, ratings_gpu, game_spec))
            if params.run_ratings_server:
                descs.append('Ratings')
                procs.append(launch_ratings_server(params_dict, ratings_gpu))
        else:
            if game_spec.reference_player_family is not None or benchmark_tag is not None:
                descs.append('Eval')
                procs.append(launch_eval_server(params_dict, ratings_gpu, game_spec))
            else:
                descs.append('Benchmark')
                procs.append(launch_benchmark_server(params_dict, ratings_gpu, game_spec))

        if params.run_ratings_server and game_spec.reference_player_family is not None:
            descs.append('Ratings')
            procs.append(launch_ratings_server(params_dict, ratings_gpu))

        loop_control_proc = procs[descs.index('Loop-controller')]
        atexit.register(subprocess_util.terminate_processes, [loop_control_proc])

        loop = True
        while loop:
            any_subprocess_error = False
            for descr, proc in zip(descs, procs):
                if proc.poll() is None:
                    continue
                loop = False
                if proc.returncode != 0:
                    any_subprocess_error = True
                    print('*' * 80)
                    logger.error('%s process %s exited with code %s', descr, proc.pid,
                                 proc.returncode)
                    print('*' * 80)
                    if proc.stderr is not None:
                        print(proc.stderr.read())
                else:
                    print('*' * 80)
                    logger.info('%s process %s exited with code %s', descr, proc.pid,
                                proc.returncode)
            time.sleep(1)

        if any_subprocess_error:
            raise subprocess.CalledProcessError(1, 'Subprocess exited with error')

    except KeyboardInterrupt:
        logger.info('run_local caught Ctrl-C')
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        logger.error('Error in subprocess: %s', e)
        if e.stderr:
            logger.error('Subprocess stderr: %s', e.stderr.decode())
        sys.exit(1)

    logger.info('All processes exited cleanly')


if __name__ == '__main__':
    main()
