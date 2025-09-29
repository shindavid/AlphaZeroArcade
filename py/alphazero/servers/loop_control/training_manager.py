from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import Domain, Generation
from alphazero.logic.game_log_reader import GameLogReader
from alphazero.logic.net_trainer import NetTrainer, TrainingStats
from alphazero.logic.sample_window_logic import Window, construct_window, get_required_dataset_size
from shared.model import Model
from shared.model_config import ModelConfig, ModelConfigGenerator
from util.py_util import make_hidden_filename

import torch
from torch import optim

import logging
import os
import shutil
import tempfile
import threading
import time
from typing import List, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, controller: LoopController):
        """
        Some members are initialized lazily in setup(). This is because those values require
        database access, and we don't want to do that in __init__.
        """
        controller.register_shutdown_action(self._shutdown, 'train-shutdown')

        self._controller = controller
        self._lock = threading.Lock()

        paradigm = controller.search_paradigm
        self._game_log_reader = GameLogReader(controller.game_spec, controller.build_params,
                                              controller.params.cuda_device, paradigm)

        self._trainer = None
        self._net = None
        self._opt = None
        self._loss_terms = None
        self._stats: Optional[TrainingStats] = None
        self._model_counts_dumped = False

        self._retraining = False
        self._checkpoint = controller.training_params.samples_per_window()  # gen-0 checkpoint
        self._last_sample_window: Optional[Window] = None  # initialized lazily
        self._latest_gen: Generation = 0
        self._train_start_ts = None

    @property
    def last_sample_window(self) -> Window:
        assert self._last_sample_window is not None
        return self._last_sample_window

    @property
    def training_params(self):
        return self._controller.training_params

    def merge_game_log_files(self, input_filenames: List[str], output_filename: str):
        self._game_log_reader.merge_game_log_files(input_filenames, output_filename)

    def get_oldest_required_gen(self) -> Generation:
        """
        Returns the generation corresponding to self.last_sample_window.start.

        The significance of this generation is that when loading a run from persistent storage to
        local storage, we only need to load data from this generation onwards.
        """
        last_sample_window = self._load_last_sample_window()

        with self._controller.self_play_db_conn_pool.db_lock:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            start = last_sample_window.start
            where_clause = f'cumulative_positions >= {start}'
            query = f'SELECT gen FROM self_play_data WHERE {where_clause} ORDER BY id LIMIT 1'
            cursor.execute(query)
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return 0
        return row[0]

    def latest_gen(self) -> Generation:
        return self._latest_gen

    def setup(self):
        """
        Performs some lazy initialization that can't be done in __init__.
        """
        self._last_sample_window = self._load_last_sample_window()
        self._latest_gen = self._controller.organizer.get_latest_model_generation(default=0)
        start = self._last_sample_window.start
        self._game_log_reader.init_data_loader(self._controller.organizer.self_play_data_dir)

        gens = []
        row_counts = []
        cumulative_row_counts = []
        file_sizes = []
        with self._controller.self_play_db_conn_pool.db_lock:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            where_clause = f'cumulative_positions >= {start}'
            query = f'SELECT gen, positions, cumulative_positions, file_size FROM self_play_data WHERE {where_clause}'
            cursor.execute(query)

            for row in cursor:
                gens.append(row[0])
                row_counts.append(row[1])
                cumulative_row_counts.append(row[2])
                file_sizes.append(row[3])
            cursor.close()

        cumulative_row_count = max(cumulative_row_counts + [0])
        self._game_log_reader.restore_data_loader(gens, row_counts, file_sizes,
                                                  cumulative_row_count)

    def retrain_models(self):
        if self._controller.organizer.fork_info is not None:
            n_retrain_gens = len(self._controller.organizer.fork_info.train_windows)
            self._retraining = True
            self.train_gen1_model_if_necessary()
            while self._latest_gen < n_retrain_gens:
                self.train_step()
            self._retraining = False

    def train_gen1_model_if_necessary(self):
        if self._latest_gen == 0:
            table: GpuContentionTable = self._controller.get_gpu_lock_table_for_training()
            table.acquire_lock(Domain.TRAINING)
            self._train_start_ts = time.time_ns()
            subgen = 0
            while True:
                subgen += 1
                if self._train_step_helper(table, subgen):
                    break
            table.release_lock(Domain.TRAINING)

    def train_step(self):
        """
        Performs a train step.

        Uses a separate thread to ensure that the DataLoader is properly cleaned up after the
        train step is complete.
        """
        table: GpuContentionTable = self._controller.get_gpu_lock_table_for_training()
        table.acquire_lock(Domain.TRAINING)
        self._train_start_ts = time.time_ns()
        self._train_step_helper(table)
        table.release_lock(Domain.TRAINING)

    def get_checkpoint(self):
        return self._checkpoint

    def estimate_upcoming_checkpoint(self):
        training_params = self.training_params
        minibatch_size = training_params.minibatch_size
        n_minibatches = training_params.minibatches_per_epoch

        f = training_params.window_size_function
        n = self._controller.get_num_committed_rows()
        w = int(f(n))
        start = max(0, n - w)
        end = n

        n_samples = minibatch_size * n_minibatches
        estimated_sample_window = construct_window(self._last_sample_window, start, end, n_samples)
        size = get_required_dataset_size(self.training_params, estimated_sample_window)
        return size

    def notify_of_new_self_play_data(self, gen: Generation, n_rows: int, file_size: int):
        self._game_log_reader.add_gen(gen, n_rows, file_size)

    def _shutdown(self):
        with self._lock:
            if self._trainer is not None:
                logger.info('Closing DataLoader...')
                self._trainer.shutdown()
            self._game_log_reader.close()

    def _load_last_sample_window(self) -> Window:
        with self._controller.training_db_conn_pool.db_lock:
            cursor = self._controller.training_db_conn_pool.get_cursor()
            cursor.execute("""SELECT window_start, window_end, window_sample_rate
                            FROM training ORDER BY gen DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            # kZero-style initialization of sample window
            # samples_per_window = self.training_params.samples_per_window()
            # target_sample_rate = self.training_params.target_sample_rate
            # return Window(0, samples_per_window, target_sample_rate)
            return Window(0, 0, 0)
        return Window(*row)

    def _load_last_checkpoint(self, model_cfg_generator_type: Type[ModelConfigGenerator]):
        organizer = self._controller.organizer
        gen = organizer.get_last_checkpointed_generation()
        if gen is None:
            return

        checkpoint_filename = organizer.get_checkpoint_filename(gen)
        logger.info('Loading checkpoint: %s', checkpoint_filename)

        # copying the checkpoint to somewhere local first seems to bypass some sort of
        # filesystem issue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.pt')
            shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
            checkpoint = torch.load(tmp_checkpoint_filename, weights_only=False)
            self._net = Model.load_from_checkpoint(checkpoint)

    def _post_init(self, model_cfg_generator_type: Type[ModelConfigGenerator]):
        """
        Performs some post-initialization that can only be performed after self._net has been
        created (either from scratch or from a checkpoint).

        - Moves self._net to cuda device and puts it in train mode.
        - Initializes self.{_net, _opt, _loss_terms}.
        """
        self._net.cuda(device=self._controller.params.cuda_device)
        self._net.train()

        self._opt = model_cfg_generator_type.optimizer(self._net.parameters())

        self._loss_terms = model_cfg_generator_type.loss_terms()
        for term in self._loss_terms:
            term.post_init(self._net)

        # Validate that network heads match c++ TrainingTargets
        logger.debug('Validating heads...')
        head_shape_info_dict = self._game_log_reader.head_shape_info_dict
        gen_cls = model_cfg_generator_type.__name__

        for h, name in enumerate(self._net.head_names):
            shape_info = head_shape_info_dict.get(name, None)
            if shape_info is None:
                raise ValueError(f'{gen_cls} heads do not match c++ TrainingTargets '
                                 f'({name} not found)')

            t = shape_info.target_index
            if h != t:
                raise ValueError(f'{gen_cls} heads do not match c++ TrainingTargets '
                                 f'({h} != {t}) for {name})')
        logger.debug('Validation complete!')

        # TODO: SWA, cyclic learning rate

    def _init_net_if_necessary(self):
        if self._net is not None:
            return

        organizer = self._controller.organizer
        checkpoint_gen = organizer.get_last_checkpointed_generation()

        game_spec = self._controller.game_spec
        head_shape_info_dict = self._game_log_reader.head_shape_info_dict
        model_cfg_generator_type = game_spec.model_configs[self._controller.params.model_cfg]
        model_cfg = model_cfg_generator_type.generate(head_shape_info_dict)

        if checkpoint_gen is None:
            self._net = Model(model_cfg)
        else:
            self._load_last_checkpoint(model_cfg_generator_type)

        self._post_init(model_cfg_generator_type)

    def _train_step_helper(self, table: GpuContentionTable, subgen=None):
        self._controller.spawn_log_sync_thread()

        gen = self._latest_gen + 1

        n_minibatches = self.training_params.minibatches_per_epoch
        trainer = NetTrainer(gen, n_minibatches, self._controller.params.cuda_device)
        with self._lock:
            self._trainer = trainer

        logger.info('******************************')
        if subgen is None:
            logger.info('Train gen:%s', gen)
        else:
            logger.info('Train gen:%s:%s', gen, subgen)
        thread = threading.Thread(target=self._do_training_epoch, name='train_step', daemon=False,
                                  args=(trainer, gen))
        thread.start()
        thread.join()

        checkpoint_set = False
        if self._stats is not None:
            self._update_window()
            if self._set_checkpoint(gen):
                checkpoint_set = True
                self._save_model(gen, self._net)
                self._record_stats(gen)
            else:
                assert subgen is not None, 'Unexpected bug'

        with self._lock:
            self._trainer = None

        self._controller.wait_for_log_sync_thread()
        self._controller.handle_new_model()
        return checkpoint_set

    def _dump_model_counts(self):
        if self._model_counts_dumped:
            return

        logger.info('Model parameter counts:')
        pcounts = self._net.get_parameter_counts()
        longest_name_len = max(len(name) for name in pcounts)
        total_count = sum(pcounts.values())
        count_len = len(str(total_count))

        #    stem:   2308 [14.7%]
        #  blocks: 206792 [82.3%]
        # ...

        fmt = f'{{:<{longest_name_len}}} : {{:>{count_len}}} [{{:>5.1f}}%]'
        for name, count in pcounts.items():
            pct = 100 * count / total_count
            logger.info(fmt.format(name, count, pct))

        self._model_counts_dumped = True

    def _do_training_epoch(self, trainer: NetTrainer, gen: Generation):
        try:
            training_params = self.training_params
            minibatch_size = training_params.minibatch_size
            n_minibatches = training_params.minibatches_per_epoch

            fork_info = self._controller.organizer.fork_info
            if fork_info is not None and gen in fork_info.train_windows:
                start, end = fork_info.train_windows[gen]
            else:
                f = training_params.window_size_function
                n = self._controller.get_num_committed_rows()
                w = int(f(n))
                logger.debug('Training window size: f(%s)=%s', n, w)

                start = max(0, n - w)
                end = n

            self._init_net_if_necessary()
            self._dump_model_counts()

            self._stats = trainer.do_training_epoch(
                self._game_log_reader, self._net, self._opt, minibatch_size, n_minibatches,
                start, end, gen, self._loss_terms)
        except:
            if self._game_log_reader.closed():
                # This is a shutdown race-condition, it's ok
                self._stats = None
                pass
            else:
                logger.error('Unexpected error in train_step():', exc_info=True)
                self._controller.request_shutdown(1)

    def _update_window(self):
        stats = self._stats

        window_start = stats.window_start
        window_end = stats.window_end
        n_samples = stats.n_samples

        window = construct_window(self._last_sample_window, window_start, window_end, n_samples)
        self._last_sample_window = window

    def _set_checkpoint(self, gen: Generation) -> bool:
        """
        Sets the next checkpoint based on the current training parameters and the last sample
        window. Returns True if the checkpoint was updated, False otherwise.
        """
        if self._retraining:
            num_committed_rows = self._controller.organizer.fork_info.train_windows[gen][1]
        else:
            num_committed_rows = self._controller.get_num_committed_rows()
        self._checkpoint = get_required_dataset_size(self.training_params, self._last_sample_window)
        logger.debug('_set_checkpoint(gen=%s) retraining:%s num-committed:%s checkpoint:%s', gen,
                     self._retraining, num_committed_rows, self._checkpoint)
        return self._checkpoint > num_committed_rows

    def _record_stats(self, gen: Generation):
        stats = self._stats
        training_params = self.training_params
        n_minibatches = training_params.minibatches_per_epoch
        minibatch_size = training_params.minibatch_size

        assert self._train_start_ts is not None
        start_ts = self._train_start_ts
        end_ts = time.time_ns()
        self._train_start_ts = None

        window = self._last_sample_window

        head_data = []
        for head_stats in stats.substats_list:
            head_name = head_stats.name
            loss = head_stats.loss()
            loss_weight = head_stats.loss_weight
            head_data.append((gen, head_name, loss, loss_weight))

        with self._controller.training_db_conn_pool.db_lock:
            conn = self._controller.training_db_conn_pool.get_connection()
            cursor = conn.cursor()

            cursor.execute("""INSERT OR REPLACE INTO training (gen, training_start_ts, training_end_ts,
                minibatch_size, n_minibatches, window_start, window_end, window_sample_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                           (gen, start_ts, end_ts, minibatch_size, n_minibatches,
                            window.start, window.end, window.sample_rate))

            cursor.executemany("""INSERT OR REPLACE INTO training_heads (gen, head_name, loss, loss_weight)
                VALUES (?, ?, ?, ?)""", head_data)

            conn.commit()
            cursor.close()

    def _save_model(self, gen: Generation, net: Model):
        organizer = self._controller.organizer
        checkpoint_filename = organizer.get_checkpoint_filename(gen)
        model_filename = organizer.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        checkpoint = {}
        net.add_to_checkpoint(checkpoint)
        torch.save(checkpoint, tmp_checkpoint_filename)

        input_shape_info_dict = self._game_log_reader.input_shape_info_dict
        head_shape_info_dict = self._game_log_reader.head_shape_info_dict
        head_names = set(head_shape_info_dict.keys())
        logger.debug('Calling save_model()...')
        net.save_model(tmp_model_filename, input_shape_info_dict, head_names)

        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        self._latest_gen = gen
        logger.debug('Checkpoint saved: %s', checkpoint_filename)
        logger.info('Model saved: %s', model_filename)
