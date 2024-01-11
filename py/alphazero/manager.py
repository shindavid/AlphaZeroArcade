"""
NOTE: in pytorch, it is standard to use the .pt extension for all sorts of files. I find this confusing. To clearly
differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
        stdout.txt
        training.db
        self-play-data/
            client-0/
                gen-0/  # uses implicit dummy uniform model
                    {timestamp}-{num_positions}.ptd
                    ...
                gen-1/  # uses models/gen-1.ptj
                    ...
                gen-2/  # uses models/gen-2.ptj
                    ...
                ...
            client-1/
                gen-3/
                    ...
                ...
            ...
        models/
            gen-1.ptj
            gen-2.ptj
            ...
        bins/
            {hash1}
            {hash2}
            ...
        checkpoints/
            gen-1.ptc
            gen-2.ptc
            ...
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Optional, List, Tuple

import torch
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader

from alphazero.cmd_server import CmdServer
from alphazero.custom_types import Generation
from alphazero.net_trainer import NetTrainer
from alphazero.optimization_args import ModelingArgs
from alphazero.sample_window_logic import SamplingParams
from games import GameType
from net_modules import Model
from util import subprocess_util
from util.py_util import timed_print, make_hidden_filename, sha256sum
from util.repo_util import Repo


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class AlphaZeroManager:
    def __init__(self, game_type: GameType, base_dir: str, binary_path: Optional[str] = None):
        self.game_type = game_type
        self.base_dir: str = base_dir
        self.log_file = None
        self.self_play_proc: Optional[subprocess.Popen] = None
        self.cmd_server: Optional[CmdServer] = None
        # self.training_db_conn: Optional[sqlite3.Connection] = None

        self.cuda_device_count = torch.cuda.device_count()
        assert self.cuda_device_count > 0, 'No cuda devices found'

        self._net = None
        self._opt = None

        self.cmd_server_db_filename = os.path.join(self.base_dir, 'cmd-server.db')
        self.stdout_filename = os.path.join(self.base_dir, 'stdout.txt')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.bins_dir = os.path.join(self.base_dir, 'bins')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')

        self._binary_path_set = False
        self._binary_path = binary_path
        self.model_cfg = None

    def set_model_cfg(self, model_cfg: str):
        self.model_cfg = model_cfg

    def makedirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.bins_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)

    def fork_from(self, manager: 'AlphaZeroManager', gen: Optional[Generation] = None):
        """
        Forks an existing run. This is accomplished by creating a bunch of soft-links to a previous
        run. Note that deleting the previous directory will break the fork. If gen is specified,
        only forks contents up-to-and-including generation gen. Otherwise, forks all contents.

        Addtionally creates a local record of the fork action in a fork.txt file. This file is used
        to short-circuit future fork operations (e.g., when the current run is restarted with the
        same cmd). The fork.txt file can also be checked by clean-up scripts to determine if a
        given base-dir is safe to remove.
        """
        assert False, 'fork_from() is currently disabled'

        fork_txt_path = os.path.join(self.base_dir, 'fork.txt')
        if os.path.isfile(fork_txt_path):
            with open(fork_txt_path, 'r') as f:
                lines = f.readlines()

            metadata = {}
            for line in lines:
                colon = line.find(':')
                if colon == -1:
                    continue
                key = line[:colon].strip()
                value = line[colon+1:].strip()
                metadata[key] = value

            g = int(metadata['Gen'])
            timed_print(f'Skipping fork from {manager.base_dir} (fork.txt already exists, at gen {g})')
            return

        def should_skip(filename, ext=None):
            if ext is not None and filename.split('.')[-1] != ext:
                return True
            if gen is None:
                return False
            return PathInfo(filename).generation > gen

        last_ts = 0

        shutil.copy(manager.stdout_filename, self.stdout_filename)
        for model_filename in os.listdir(manager.models_dir):
            if should_skip(model_filename, 'ptj'):
                continue
            src = os.path.join(manager.models_dir, model_filename)
            tgt = os.path.join(self.models_dir, model_filename)
            os.symlink(src, tgt)
            last_ts = max(last_ts, os.path.getmtime(src))

        for checkpoint_filename in os.listdir(manager.checkpoints_dir):
            if should_skip(checkpoint_filename, 'ptc'):
                continue
            src = os.path.join(manager.checkpoints_dir, checkpoint_filename)
            tgt = os.path.join(self.checkpoints_dir, checkpoint_filename)
            os.symlink(src, tgt)
            last_ts = max(last_ts, os.path.getmtime(src))

        for self_play_subdir in os.listdir(manager.self_play_data_dir):
            if should_skip(self_play_subdir):
                continue
            src = os.path.join(manager.self_play_data_dir, self_play_subdir)
            tgt = os.path.join(self.self_play_data_dir, self_play_subdir)
            os.symlink(src, tgt)
            last_ts = max(last_ts, os.path.getmtime(src))

        # only copy bins that were modified after the last timestamp, to ensure that the forked
        # run continue with the same binary that was used up to {gen} in the original run
        for bin_filename in os.listdir(manager.bins_dir):
            src = os.path.join(manager.bins_dir, bin_filename)
            if os.path.getmtime(src) > last_ts:
                continue
            tgt = os.path.join(self.bins_dir, bin_filename)
            os.symlink(src, tgt)

        # copy the ratings.db file if it exists
        original_ratings_db_filename = os.path.join(manager.base_dir, 'ratings.db')
        if os.path.isfile(original_ratings_db_filename):
            shutil.copy(original_ratings_db_filename, self.base_dir)

            if gen is not None:
                # erase db contents after gen.
                #
                # TODO: move this into a separate file along with the relevant parts of
                # compute-ratings.py
                ratings_db_filename = os.path.join(self.base_dir, 'ratings.db')
                conn = sqlite3.connect(ratings_db_filename)
                c = conn.cursor()
                for table in ('matches', 'ratings', 'x_values'):
                    c.execute(f'DELETE FROM {table} WHERE mcts_gen > ?', (gen,))
                conn.commit()
                conn.close()

        forked_gen = gen if gen is not None else manager.get_latest_generation()
        with open(fork_txt_path, 'w') as f:
            f.write(f'From: {manager.base_dir}\n')
            f.write(f'Gen: {forked_gen}\n')

        self.init_logging(self.stdout_filename)
        timed_print(f'Forked from {manager.base_dir} (gen: {forked_gen})')

    def copy_binary(self, bin_src):
        bin_md5 = str(sha256sum(bin_src))
        bin_tgt = os.path.join(self.bins_dir, bin_md5)
        rsync_cmd = ['rsync', '-t', bin_src, bin_tgt]
        subprocess_util.run(rsync_cmd)
        return bin_tgt

    @property
    def binary_path(self):
        if self._binary_path_set:
            return self._binary_path

        self._binary_path_set = True
        if self._binary_path:
            bin_tgt = self.copy_binary(self._binary_path)
            timed_print(f'Using cmdline-specified binary {self._binary_path} (copied to {bin_tgt})')
            self._binary_path = bin_tgt
            return self._binary_path

        candidates = os.listdir(self.bins_dir)
        if len(candidates) == 0:
            bin_name = self.game_type.binary_name
            bin_src = os.path.join(Repo.root(), f'target/Release/bin/{bin_name}')
            bin_tgt = self.copy_binary(bin_src)
            self._binary_path = bin_tgt
            timed_print(f'Using binary {bin_src} (copied to {bin_tgt})')
        else:
            # get the candidate with the most recent mtime:
            candidates = [os.path.join(self.bins_dir, c) for c in candidates]
            candidates = [(os.path.getmtime(c), c) for c in candidates]
            candidates.sort()
            bin_tgt = candidates[-1][1]
            self._binary_path = bin_tgt
            timed_print(f'Using most-recently used binary: {bin_tgt}')

        return self._binary_path

    def erase_data_after(self, gen: Generation):
        """
        Deletes self-play/ dirs strictly greater than gen, and all models/checkpoints trained off those dirs
        (i.e., models/checkpoints strictly greater than gen+1).
        """
        assert False, 'erase_data_after() is currently disabled'
        timed_print(f'Erasing data after gen {gen}')
        g = gen + 1
        while True:
            gen_dir = os.path.join(self.self_play_data_dir, f'gen-{g}')
            if os.path.exists(gen_dir):
                shutil.rmtree(gen_dir, ignore_errors=True)
                g += 1
            else:
                break

        g = gen + 2
        while True:
            model = os.path.join(self.models_dir, f'gen-{g}.ptj')
            checkpoint = os.path.join(self.checkpoints_dir, f'gen-{g}.ptc')
            found = False
            for f in [model, checkpoint]:
                if os.path.exists(f):
                    os.remove(f)
                    found = True
            if not found:
                break
            g += 1

    def load_last_checkpoint(self):
        """
        If a prior checkpoint exists, does the following:

        - Sets self._net
        - Sets self._opt
        - Sets self.dataset_generator.expected_sample_counts
        """
        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            return

        gen = checkpoint_info.generation
        checkpoint_filename = self.get_checkpoint_filename(gen)
        timed_print(f'Loading checkpoint: {checkpoint_filename}')

        # copying the checkpoint to somewhere local first seems to bypass some sort of
        # filesystem issue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
            shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
            checkpoint = torch.load(tmp_checkpoint_filename)
            self._net = Model.load_from_checkpoint(checkpoint)

        self._init_net_and_opt()

    def _init_net_and_opt(self):
        """
        Assumes that self._net has been initialized, and that self._opt has not.

        Moves self._net to cuda device and puts it in train mode.

        Initializes self._opt.
        """
        self._net.cuda(device=0)  # net training always uses device 0
        self._net.train()

        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

    def get_net_and_optimizer(self, loader: 'DataLoader') -> Tuple[Model, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            input_shape = loader.dataset.get_input_shape()
            target_names = loader.dataset.get_target_names()
            self._net = Model(self.game_type.model_dict[self.model_cfg](input_shape))
            self._net.validate_targets(target_names)
            self._init_net_and_opt()
            timed_print(f'Creating new net with input shape {input_shape}')
        else:
            self.load_last_checkpoint()

        return self._net, self._opt

    def init_logging(self, filename: str):
        if self.log_file is not None:
            return
        self.log_file = open(filename, 'a')
        sys.stdout = self
        sys.stderr = self

    def write(self, msg):
        sys.__stdout__.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)
        self.flush()

    def flush(self):
        sys.__stdout__.flush()
        if self.log_file is not None:
            self.log_file.flush()

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen-{gen}.ptj')

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}.ptc')

    @staticmethod
    def get_ordered_subpaths(path: str) -> List[str]:
        subpaths = list(natsorted(f for f in os.listdir(path)))
        return [f for f in subpaths if not f.startswith('.')]

    @staticmethod
    def get_latest_full_subpath(path: str) -> Optional[str]:
        subpaths = AlphaZeroManager.get_ordered_subpaths(path)
        return os.path.join(path, subpaths[-1]) if subpaths else None

    @staticmethod
    def get_latest_info(path: str) -> Optional[PathInfo]:
        subpaths = AlphaZeroManager.get_ordered_subpaths(path)
        if not subpaths:
            return None
        return PathInfo(subpaths[-1])

    def get_latest_model_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.models_dir)

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return AlphaZeroManager.get_latest_info(self.checkpoints_dir)

    def get_latest_model_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.models_dir)
        return 0 if info is None else info.generation

    def get_latest_generation(self) -> Generation:
        return self.get_latest_model_generation()

    def get_latest_model_filename(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.models_dir)

    def run_gen0_if_necessary(self):
        """
        Runs a single self-play generation using the dummy uniform model.

        If the gen-0 self-play data already exists, does nothing.

        TODO: change max-rows-reached detection mechanism to be controlled outside of the
        cmd-server.
        """
        if self.cmd_server.is_gen0_complete():
            return

        max_rows = self.cmd_server.n_samples_per_window

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--games-base-dir', self.self_play_data_dir,
            '--do-not-report-metrics',
            '--max-rows', max_rows,

            # for gen-0, sample more positions and use fewer iters per game, so we finish faster
            '--num-full-iters', 100,
            '--full-pct', 1.0,
        ]
        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--cmd-server-port', self.cmd_server.port,
            '--starting-generation', 0,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        timed_print(f'Running gen-0 self-play [{proc.pid}]: {self_play_cmd}')
        subprocess_util.wait_for(proc)

        self.cmd_server.wait_for_client_disconnect()
        assert len(self.cmd_server.get_clients_list()) == 0

        timed_print('Gen-0 self-play complete')

    def train_gen1_model_if_necessary(self):
        """
        Trains a single model using the gen-0 self-play data.

        If the gen-1 model already exists, does nothing.
        """
        gen = 1
        model_filename = self.get_model_filename(gen)
        if os.path.isfile(model_filename):
            return

        self.train_step()

    def launch_cmd_server(self, sampling_params: SamplingParams, port):
        self.cmd_server = CmdServer(sampling_params, self.base_dir, port=port)
        self.cmd_server.start()

    def launch_self_play(self):
        cuda_device_id = self.cuda_device_count - 1
        cuda_device = f'cuda:{cuda_device_id}'
        shared_gpu = cuda_device_id == 0

        gen = self.get_latest_model_generation()
        model_filename = self.get_model_filename(gen)

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '-D', self.self_play_data_dir,
            '-m', model_filename,
            '--cuda-device', cuda_device,
        ]
        if shared_gpu:
            player_args.append('--shared-gpu')

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        bin_tgt = self.binary_path

        self_play_cmd = [
            bin_tgt,
            '-G', 0,
            '--cmd-server-port', self.cmd_server.port,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))
        stdout = open(os.path.join(self.base_dir, 'self-play.stdout'), 'a')
        stderr = open(os.path.join(self.base_dir, 'self-play.stderr'), 'a')
        self.self_play_proc = subprocess_util.Popen(self_play_cmd, stdout=stdout, stderr=stderr)

    def wait_until_enough_training_data(self):
        self.cmd_server.wait_until_enough_training_data()

    def train_step(self):
        gen = self.get_latest_model_generation() + 1

        dataset = self.cmd_server.get_position_dataset()

        print('******************************')
        timed_print(f'Train gen:{gen}')
        dataset.announce_sampling(timed_print)

        trainer = NetTrainer(gen, ModelingArgs.snapshot_steps)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=ModelingArgs.minibatch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True)

        net, optimizer = self.get_net_and_optimizer(loader)

        self.cmd_server.pause_shared_gpu_clients()

        stats = trainer.do_training_epoch(loader, net, optimizer, dataset)
        stats.dump()

        self.cmd_server.record_training_step(stats)
        assert trainer.n_minibatches_processed >= ModelingArgs.snapshot_steps

        timed_print(f'Gen {gen} training complete')
        trainer.dump_timing_stats()

        checkpoint_filename = self.get_checkpoint_filename(gen)
        model_filename = self.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        checkpoint = {}
        net.add_to_checkpoint(checkpoint)
        torch.save(checkpoint, tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        timed_print(f'Checkpoint saved: {checkpoint_filename}')
        timed_print(f'Model saved: {model_filename}')

        if gen > 1:
            self.cmd_server.reload_weights(model_filename, gen)

    def run(self):
        self.init_logging(self.stdout_filename)
        self.run_gen0_if_necessary()
        self.train_gen1_model_if_necessary()
        self.load_last_checkpoint()
        self.launch_self_play()

        while True:
            self.wait_until_enough_training_data()
            self.train_step()
