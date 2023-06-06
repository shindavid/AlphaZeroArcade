"""
NOTE: in pytorch, it is standard to use the .pt extension for all sorts of files. I find this confused. To clearly
differentiate the different types of files, I have invented the following extensions:

- .ptd: pytorch-data files
- .ptc: pytorch-checkpoint files
- .ptj: pytorch-jit-compiled model files

BASE_DIR/
         stdout.txt
         self-play-data/
             gen-0/
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
import signal
import sys
import tempfile
import time
from typing import Optional, List, Tuple

import torch
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader

from alphazero.custom_types import Generation
from alphazero.data.games_dataset import GamesDataset
from alphazero.optimization_args import ModelingArgs
from games import GameType
from neural_net import NeuralNet, LearningTarget
from util import subprocess_util
from util.py_util import timed_print, make_hidden_filename, sha256sum
from util.repo_util import Repo
from util.torch_util import apply_mask


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class SelfPlayProcData:
    def __init__(self, cmd: str, n_games: int, gen: Generation, games_dir: str):
        self.proc_complete = False
        self.proc = subprocess_util.Popen(cmd)
        self.n_games = n_games
        self.gen = gen
        self.games_dir = games_dir
        timed_print(f'Running gen-{gen} self-play [{self.proc.pid}]: {cmd}')

        if self.n_games:
            self.wait_for_completion()

    def terminate(self, timeout: Optional[int] = None):
        if self.proc_complete:
            return
        self.proc.kill()
        self.wait_for_completion(timeout=timeout, expected_return_code=-int(signal.SIGKILL))

    def wait_for_completion(self, timeout: Optional[int] = None, expected_return_code: int = 0):
        subprocess_util.wait_for(self.proc, timeout=timeout, expected_return_code=expected_return_code)
        AlphaZeroManager.finalize_games_dir(self.games_dir)
        timed_print(f'Completed gen-{self.gen} self-play [{self.proc.pid}]')
        self.proc_complete = True


class AlphaZeroManager:
    def __init__(self, game_type: GameType, base_dir: str):
        self.game_type = game_type
        self.py_cuda_device: int = 0
        self.log_file = None

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

    def makedirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.bins_dir, exist_ok=True)
        os.makedirs(self.players_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)

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

    def get_net_and_optimizer(self, loader: 'DataLoader') -> Tuple[NeuralNet, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            input_shape = loader.dataset.get_input_shape()
            target_names = loader.dataset.get_target_names()
            self._net = self.game_type.net_type.create(input_shape, target_names)
            timed_print(f'Creating new net with input shape {input_shape}')
        else:
            gen = checkpoint_info.generation
            checkpoint_filename = self.get_checkpoint_filename(gen)
            timed_print(f'Loading checkpoint: {checkpoint_filename}')

            # copying the checkpoint to somewhere local first seems to bypass some sort of filesystem issue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
                shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
                self._net = self.game_type.net_type.load_checkpoint(tmp_checkpoint_filename)

        self._net.cuda(device=self.py_cuda_device)
        self._net.train()

        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

        return self._net, self._opt

    def init_logging(self, filename: str):
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
            return f.read().strip()

    def get_self_play_proc(self, async_mode: bool) -> SelfPlayProcData:
        gen = self.get_latest_model_generation()

        games_dir = self.get_self_play_data_subdir(gen)
        bin_name = self.game_type.binary_name
        bin_src = os.path.join(Repo.root(), f'target/Release/bin/{bin_name}')
        bin_md5 = str(sha256sum(bin_src))
        bin_tgt = os.path.join(self.bins_dir, bin_md5)
        rsync_cmd = ['rsync', '-t', bin_src, bin_tgt]
        subprocess_util.run(rsync_cmd)

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
        if gen:
            player_args.append('--no-clear-dir')

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            bin_tgt,
            '-G', n_games,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        competitive_player_args = ['--type=MCTS-C'] + base_player_args
        competitive_player_str = '%s --player "%s"\n' % (bin_tgt, ' '.join(map(str, competitive_player_args)))
        player_filename = os.path.join(self.players_dir, f'gen-{gen}.txt')
        with open(player_filename, 'w') as f:
            f.write(competitive_player_str)

        self_play_cmd = ' '.join(map(str, self_play_cmd))
        return SelfPlayProcData(self_play_cmd, n_games, gen, games_dir)

    def train_step(self):
        print('******************************')
        gen = self.get_latest_model_generation() + 1
        timed_print(f'Train gen:{gen}')

        for_loop_time = 0
        t0 = time.time()
        steps = 0
        while True:
            games_dataset = GamesDataset(self.self_play_data_dir)
            loader = torch.utils.data.DataLoader(
                games_dataset,
                batch_size=ModelingArgs.minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)
            assert games_dataset.n_total_games >= self.n_gen0_games

            net, optimizer = self.get_net_and_optimizer(loader)
            games_dataset.set_key_order(net.target_names())

            loss_fns = [target.loss_fn() for target in net.learning_targets]

            timed_print(f'Sampling from the {games_dataset.n_window} most recent positions among '
                        f'{games_dataset.n_total_positions} total positions (minibatches processed: {steps})')

            stats = TrainingStats(net)
            for data in loader:
                t1 = time.time()
                inputs = data[0]
                labels_list = data[1:]
                inputs = inputs.type(torch.float32).to(self.py_cuda_device_str)

                labels_list = [labels.to(self.py_cuda_device_str) for labels in labels_list]

                optimizer.zero_grad()
                outputs_list = net(inputs)
                assert len(outputs_list) == len(labels_list)

                labels_list = [labels.reshape((labels.shape[0], -1)) for labels in labels_list]
                outputs_list = [outputs.reshape((outputs.shape[0], -1)) for outputs in outputs_list]

                masks = [target.get_mask(labels) for labels, target in zip(labels_list, net.learning_targets)]

                labels_list = [apply_mask(labels, mask) for mask, labels in zip(masks, labels_list)]
                outputs_list = [apply_mask(outputs, mask) for mask, outputs in zip(masks, outputs_list)]

                loss_list = [loss_fn(outputs, labels) for loss_fn, outputs, labels in
                             zip(loss_fns, outputs_list, labels_list)]

                loss = sum([loss * target.loss_weight for loss, target in zip(loss_list, net.learning_targets)])

                results_list = [EvaluationResults(labels, outputs, loss) for labels, outputs, loss in
                                zip(labels_list, outputs_list, loss_list)]

                stats.update(results_list)

                loss.backward()
                optimizer.step()
                steps += 1
                t2 = time.time()
                for_loop_time += t2 - t1
                if steps == ModelingArgs.snapshot_steps:
                    break
            stats.dump()
            if steps >= ModelingArgs.snapshot_steps:
                break

        t3 = time.time()
        total_time = t3 - t0
        data_loading_time = total_time - for_loop_time

        timed_print(f'Gen {gen} training complete ({steps} minibatch updates)')
        timed_print(f'Data loading time: {data_loading_time:10.3f} seconds')
        timed_print(f'Training time:     {for_loop_time:10.3f} seconds')

        checkpoint_filename = self.get_checkpoint_filename(gen)
        model_filename = self.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        net.save_checkpoint(tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        timed_print(f'Checkpoint saved: {checkpoint_filename}')
        timed_print(f'Model saved: {model_filename}')

    @staticmethod
    def finalize_games_dir(games_dir: str):
        n_positions = 0
        n_games = 0
        for filename in os.listdir(games_dir):
            n = int(filename.split('-')[1].split('.')[0])
            n_positions += n
            n_games += 1

        done_file = os.path.join(games_dir, 'done.txt')
        with open(done_file, 'w') as f:
            f.write(f'n_games={n_games}\n')
            f.write(f'n_positions={n_positions}\n')
            f.write(f'done\n')

    def run(self, async_mode: bool = True):
        if async_mode:
            """
            TODO: assert that 2 GPU's are actually available.
            TODO: make this configurable, this is specific to dshin's setup
            """
            self.py_cuda_device = 1

        self.init_logging(self.stdout_filename)
        while True:
            self_play_proc_data = self.get_self_play_proc(async_mode)
            self.train_step()
            self_play_proc_data.terminate(timeout=300)


class EvaluationResults:
    def __init__(self, labels, outputs, loss):
        self.labels = labels
        self.outputs = outputs
        self.loss = loss

    def __len__(self):
        return len(self.labels)


class TrainingSubStats:
    max_descr_len = 0

    def __init__(self, target: LearningTarget):
        self.target = target
        self.accuracy_num = 0.0
        self.loss_num = 0.0
        self.den = 0

        TrainingSubStats.max_descr_len = max(TrainingSubStats.max_descr_len, len(self.descr))

    @property
    def descr(self) -> str:
        return self.target.name

    def update(self, results: EvaluationResults):
        n = len(results)
        self.accuracy_num += self.target.get_num_correct_predictions(results.outputs, results.labels)
        self.loss_num += float(results.loss.item()) * n
        self.den += n

    def accuracy(self):
        return self.accuracy_num / self.den if self.den else 0.0

    def loss(self):
        return self.loss_num / self.den if self.den else 0.0

    def dump(self):
        tuples = [
            (' accuracy:', self.accuracy()),
            (' loss:', self.loss()),
        ]
        max_str_len = max([len(t[0]) for t in tuples]) + TrainingSubStats.max_descr_len
        for key, value in tuples:
            full_key = self.descr + key
            print(f'{full_key.ljust(max_str_len)} %8.6f' % value)


class TrainingStats:
    def __init__(self, net: NeuralNet):
        self.substats_list = [TrainingSubStats(target) for target in net.learning_targets]

    def update(self, results_list: List[EvaluationResults]):
        for results, substats in zip(results_list, self.substats_list):
            substats.update(results)

    def dump(self):
        for substats in self.substats_list:
            substats.dump()
