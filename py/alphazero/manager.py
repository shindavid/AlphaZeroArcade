"""
NOTE: in pytorch, it is standard to use the .pt extension for all sorts of files. I find this confusing. To clearly
differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         stdout.txt
         self-play-data/
             gen-0/
                 kill.txt  # marker to communicate stop signal to self-play process
                 done.txt  # written after gen is complete
                 {timestamp}-{num_positions}.ptd
                 ...
             gen-1/
                 ...
             gen-2/
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
         players/
             gen-1.txt  # bin --player "--type=... ..."
             ...
         checkpoints/
             gen-1.ptc
             gen-2.ptc
             ...
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from typing import Optional, List, Tuple, Callable

import torch
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader

from alphazero.custom_types import Generation
from alphazero.data.games_dataset import GamesDataset
from alphazero.net_trainer import NetTrainer
from alphazero.optimization_args import ModelingArgs
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


class SelfPlayResults:
    def __init__(self, stdout: str):
        """
        Parallelism factor:           25
        Num games:                    25
        Total runtime:            3.220s
        Avg runtime:              0.129s
        MCTS evaluated positions:   6654
        MCTS batches evaluated:      139
        MCTS avg batch size:       47.87
        """
        mappings = {}
        for line in stdout.splitlines():
            colon = line.find(':')
            if colon == -1:
                continue
            key = line[:colon].strip()
            value = line[colon+1:].strip()
            mappings[key] = value

        self.num_games = int(mappings['Num games'])
        self.total_runtime = float(mappings['Total runtime'].split('s')[0])
        self.mcts_evaluated_positions = int(mappings['MCTS evaluated positions'])
        self.mcts_batches_evaluated = int(mappings['MCTS batches evaluated'])


class SelfPlayProcData:
    def __init__(self, cmd: str, n_games: int, gen: Generation, games_dir: str):
        if os.path.exists(games_dir):
            # This likely means that we are resuming a previous run that already wrote some games to this directory.
            # In principle, we could make use of those games. However, that complicates the tracking of some stats, like
            # the total amount of time spent on self-play. Since not a lot of compute/time is spent on each generation,
            # we just blow away the directory to make our lives simpler
            if os.path.islink(games_dir):
                # remove the sym link
                os.remove(games_dir)
            else:
                shutil.rmtree(games_dir)

        self.proc_complete = False
        self.proc = subprocess_util.Popen(cmd)
        self.n_games = n_games
        self.gen = gen
        self.games_dir = games_dir
        timed_print(f'Running gen-{gen} self-play [{self.proc.pid}]: {cmd}')

        if self.n_games:
            self.wait_for_completion()

    def terminate(self, timeout: Optional[int] = None, finalize_games_dir=True,
                  expected_return_code: Optional[int] = 0):
        if self.proc_complete:
            return
        kill_file = os.path.join(self.games_dir, 'kill.txt')
        os.system(f'touch {kill_file}')  # signals c++ process to stop
        self.wait_for_completion(timeout=timeout, finalize_games_dir=finalize_games_dir,
                                 expected_return_code=expected_return_code)

    def wait_for_completion(self, timeout: Optional[int] = None, finalize_games_dir=True,
                            expected_return_code: Optional[int] = 0):
        timed_print(f'Waiting for self-play proc [{self.proc.pid}] to complete...')
        stdout = subprocess_util.wait_for(self.proc, timeout=timeout, expected_return_code=expected_return_code)
        results = SelfPlayResults(stdout)
        if finalize_games_dir:
            AlphaZeroManager.finalize_games_dir(self.games_dir, results)
        timed_print(f'Completed gen-{self.gen} self-play [{self.proc.pid}]')
        self.proc_complete = True


class AlphaZeroManager:
    def __init__(self, game_type: GameType, base_dir: str, binary_path: Optional[str] = None):
        self.game_type = game_type
        self.py_cuda_device: int = 0
        self.log_file = None
        self.self_play_proc_data: Optional[SelfPlayProcData] = None

        self.n_gen0_games = 4000
        self.n_sync_games = 1000
        self.base_dir: str = base_dir

        self._net = None
        self._opt = None

        self.stdout_filename = os.path.join(self.base_dir, 'stdout.txt')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.bins_dir = os.path.join(self.base_dir, 'bins')
        self.players_dir = os.path.join(self.base_dir, 'players')
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
        os.makedirs(self.players_dir, exist_ok=True)
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

        for player_filename in os.listdir(manager.players_dir):
            if should_skip(player_filename, 'txt'):
                continue
            src = os.path.join(manager.players_dir, player_filename)
            tgt = os.path.join(self.players_dir, player_filename)
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

    @property
    def py_cuda_device_str(self) -> str:
        return f'cuda:{self.py_cuda_device}'

    def erase_data_after(self, gen: Generation):
        """
        Deletes self-play/ dirs strictly greater than gen, and all models/checkpoints trained off those dirs
        (i.e., models/checkpoints strictly greater than gen+1).
        """
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

    def get_net_and_optimizer(self, loader: 'DataLoader') -> Tuple[Model, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            input_shape = loader.dataset.get_input_shape()
            target_names = loader.dataset.get_target_names()
            self._net = Model(self.game_type.model_dict[self.model_cfg](input_shape))
            self._net.validate_targets(target_names)
            timed_print(f'Creating new net with input shape {input_shape}')
        else:
            gen = checkpoint_info.generation
            checkpoint_filename = self.get_checkpoint_filename(gen)
            timed_print(f'Loading checkpoint: {checkpoint_filename}')

            # copying the checkpoint to somewhere local first seems to bypass some sort of filesystem issue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
                shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
                checkpoint = torch.load(tmp_checkpoint_filename)
                self._net = Model.load_from_checkpoint(checkpoint)

        self._net.cuda(device=self.py_cuda_device)
        self._net.train()

        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

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

    def get_self_play_data_subdir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}')

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

    def get_latest_player_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.players_dir)
        return 0 if info is None else info.generation

    def get_latest_generation(self) -> Generation:
        return min(self.get_latest_model_generation(), self.get_latest_player_generation())

    def get_latest_self_play_data_generation(self) -> Generation:
        info = AlphaZeroManager.get_latest_info(self.self_play_data_dir)
        return 0 if info is None else info.generation

    def get_latest_model_filename(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.models_dir)

    def get_latest_self_play_data_subdir(self) -> Optional[str]:
        return AlphaZeroManager.get_latest_full_subpath(self.self_play_data_dir)

    def get_player_cmd(self, gen: Generation) -> Optional[str]:
        filename = os.path.join(self.players_dir, f'gen-{gen}.txt')
        if not os.path.exists(filename):
            return None
        with open(filename, 'r') as f:
            cmd = f.read().strip()

        return cmd

    def get_self_play_proc(self, async_mode: bool) -> SelfPlayProcData:
        gen = self.get_latest_model_generation()

        games_dir = self.get_self_play_data_subdir(gen)

        if gen == 0:
            n_games = self.n_gen0_games
        elif not async_mode:
            n_games = self.n_sync_games
        else:
            n_games = 0

        base_player_args = []  # '--no-forced-playouts']
        if gen:
            model = self.get_model_filename(gen)
            base_player_args.extend(['-m', model])

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '-g', games_dir,
        ] + base_player_args

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        bin_tgt = self.binary_path
        self_play_cmd = [
            bin_tgt,
            '-G', n_games,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        if n_games == 0:
            kill_file = os.path.join(games_dir, 'kill.txt')
            self_play_cmd.extend([
                '--kill-file', kill_file
            ])

        competitive_player_args = ['--type=MCTS-C'] + base_player_args
        competitive_player_str = '%s --player "%s"\n' % (bin_tgt, ' '.join(map(str, competitive_player_args)))
        player_filename = os.path.join(self.players_dir, f'gen-{gen}.txt')
        with open(player_filename, 'w') as f:
            f.write(competitive_player_str)

        self_play_cmd = ' '.join(map(str, self_play_cmd))
        return SelfPlayProcData(self_play_cmd, n_games, gen, games_dir)

    def train_step(self, pre_commit_func: Optional[Callable[[], None]] = None):
        """
        Performs a train-step. This performs N minibatch-updates of size S, where:

        N = ModelingArgs.snapshot_steps
        S = ModelingArgs.minibatch_size

        After the last minibatch update is complete, but before the model is committed to disk, pre_commit_func() is
        called. We use this function shutdown the c++ self-play process. This is necessary because the self-play process
        prints important metadata to stdout, and we don't want to commit a model for which we don't have the metadata.
        """
        print('******************************')
        gen = self.get_latest_model_generation() + 1
        timed_print(f'Train gen:{gen}')

        trainer = NetTrainer(ModelingArgs.snapshot_steps, self.py_cuda_device_str)
        while True:
            games_dataset = GamesDataset(self.self_play_data_dir)
            loader = torch.utils.data.DataLoader(
                games_dataset,
                batch_size=ModelingArgs.minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)

            net, optimizer = self.get_net_and_optimizer(loader)

            stats = trainer.do_training_epoch(loader, net, optimizer, games_dataset)
            stats.dump()
            if trainer.n_minibatches_processed >= ModelingArgs.snapshot_steps:
                break

        timed_print(f'Gen {gen} training complete ({trainer.n_minibatches_processed} minibatch updates)')
        trainer.dump_timing_stats()

        if pre_commit_func:
            pre_commit_func()

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

    @staticmethod
    def finalize_games_dir(games_dir: str, results: SelfPlayResults):
        timed_print('Finalizing games dir: %s' % games_dir)
        n_positions = 0
        n_games = 0
        for filename in os.listdir(games_dir):
            if filename.endswith('.txt'):
                continue
            try:
                n = int(filename.split('-')[1].split('.')[0])
            except:
                raise Exception('Could not parse filename: %s in %s' % (filename, games_dir))
            n_positions += n
            n_games += 1

        done_file = os.path.join(games_dir, 'done.txt')
        tmp_done_file = make_hidden_filename(done_file)
        with open(tmp_done_file, 'w') as f:
            f.write(f'n_games={n_games}\n')
            f.write(f'n_positions={n_positions}\n')
            f.write(f'runtime={results.total_runtime}\n')
            f.write(f'n_evaluated_positions={results.mcts_evaluated_positions}\n')
            f.write(f'n_batches_evaluated={results.mcts_batches_evaluated}\n')
        os.rename(tmp_done_file, done_file)

    def run(self, async_mode: bool = True):
        if async_mode:
            """
            TODO: assert that 2 GPU's are actually available.
            TODO: make this configurable, this is specific to dshin's setup
            """
            self.py_cuda_device = 1

        self.init_logging(self.stdout_filename)
        while True:
            self.self_play_proc_data = self.get_self_play_proc(async_mode)
            self.train_step(pre_commit_func=lambda: self.self_play_proc_data.terminate(timeout=300))

    def shutdown(self):
        """
        If there is an active self-play process, kill it, without finalizing games dir (so that on a restart, it
        continues on the same generation).
        """
        if self.self_play_proc_data is not None:
            self.self_play_proc_data.terminate(timeout=300, finalize_games_dir=False, expected_return_code=None)
            self.self_play_proc_data = None
