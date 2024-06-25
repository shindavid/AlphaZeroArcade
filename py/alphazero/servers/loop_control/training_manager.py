from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import Domain, Generation
from alphazero.logic.game_log_reader import GameLogReader
from alphazero.logic.net_trainer import NetTrainer, TrainingStats
from alphazero.logic.position_dataset import PositionDataset, PositionListSlice
from alphazero.logic.sample_window_logic import Window, construct_window, get_required_dataset_size
from shared.net_modules import Model
from util.logging_util import get_logger
from util.py_util import make_hidden_filename

import os
import shutil
import tempfile
import threading
from typing import Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader


logger = get_logger()


class TrainingManager:
    def __init__(self, controller: LoopControllerInterface):
        """
        Some members are initialized lazily in setup(). This is because those values require
        database access, and we don't want to do that in __init__.
        """
        controller.register_shutdown_action(self._shutdown)

        self._controller = controller
        self._ready_event = threading.Event()
        self._lock = threading.Lock()

        self._game_log_reader = GameLogReader(controller.game_spec)

        self._trainer = None
        self._net = None
        self._opt = None

        # This is the length that the master_list needs to be before we can start a new train loop.
        self._master_list_length_for_next_train_loop: int = 0  # initialized lazily

        self._last_sample_window: Optional[Window] = None  # initialized lazily

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length: Optional[int] = None

        self._latest_gen: Generation = 0

        self._master_list_slice = PositionListSlice()

    def latest_gen(self) -> Generation:
        return self._latest_gen

    def setup(self):
        """
        Performs some lazy initialization that can't be done in __init__.
        """
        self._last_sample_window = self._load_last_sample_window()
        self._master_list_length = self._fetch_num_total_augmented_positions()
        self._latest_gen = self._controller.organizer.get_latest_model_generation()

        if self._controller.organizer.fork_info is not None:
            max_forked_client_id = self._controller.organizer.fork_info.max_client_id
            self._master_list_slice.set_max_forked_client_id(max_forked_client_id)

    def wait_until_enough_training_data(self):
        training_params = self._controller.training_params
        with self._lock:
            self._master_list_length_for_next_train_loop = get_required_dataset_size(
                training_params, self._last_sample_window)
            logger.info(f'Waiting for more training data... (current={self._master_list_length}, '
                        f'needed={self._master_list_length_for_next_train_loop})')
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                return
            self._ready_event.clear()

        # TODO: progress-bar (use module tqdm)
        self._ready_event.wait()

    def retrain_models(self):
        if self._controller.organizer.fork_info is not None:
            n_retrain_gens = len(self._controller.organizer.fork_info.train_windows)
            while self._latest_gen < n_retrain_gens:
                self.train_step(retrain_from_fork=True)

    def train_gen1_model_if_necessary(self):
        if self._latest_gen == 0:
            self.train_step()

    def train_step(self, retrain_from_fork=False):
        """
        Performs a train step.

        Uses a separate thread to ensure that the DataLoader is properly cleaned up after the
        train step is complete.
        """
        organizer = self._controller.organizer
        fork_info = organizer.fork_info

        forked_base_dir = None if fork_info is None else fork_info.forked_base_dir
        gen = organizer.get_latest_model_generation() + 1
        if retrain_from_fork:
            self._extend_master_list_from_fork(gen)
        else:
            self._extend_master_list()

        dataset = PositionDataset(organizer.base_dir, forked_base_dir, self._master_list_slice,
                                  self._game_log_reader)

        logger.info('******************************')
        logger.info(f'Train gen:{gen}')
        dataset.announce_sampling(logger.info)

        n_minibatches = self._controller.training_params.minibatches_per_epoch

        trainer = NetTrainer(gen, n_minibatches, self._controller.params.cuda_device)
        with self._lock:
            self._trainer = trainer

        thread = threading.Thread(target=self._train_step_helper, name='train_step', daemon=False,
                                  args=(dataset, trainer, gen))
        thread.start()
        thread.join()

        with self._lock:
            self._trainer = None

    def handle_new_self_play_positions(self, n_augmented_positions: int):
        with self._lock:
            self._master_list_length += n_augmented_positions
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                self._ready_event.set()

    def _shutdown(self):
        with self._lock:
            if self._trainer is not None:
                logger.info('Closing DataLoader...')
                self._trainer.shutdown()

    def _load_last_sample_window(self) -> Window:
        with self._controller.training_db_conn_pool.db_lock:
            cursor = self._controller.training_db_conn_pool.get_cursor()
            cursor.execute("""SELECT window_start, window_end, window_sample_rate
                            FROM training ORDER BY gen DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            # kZero-style initialization of sample window
            # samples_per_window = self._controller.training_params.samples_per_window()
            # target_sample_rate = self._controller.training_params.target_sample_rate
            # return Window(0, samples_per_window, target_sample_rate)
            return Window(0, 0, 0)
        return Window(*row)

    def _fetch_num_total_augmented_positions(self) -> int:
        with self._controller.self_play_db_conn_pool.db_lock:
            # Return cumulative_augmented_positions for the last row of games:
            cursor = self._controller.self_play_db_conn_pool.get_cursor()
            cursor.execute("""SELECT cumulative_augmented_positions FROM games
                           ORDER BY id DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is not None:
            return row[0]

        return 0

    def _extend_master_list(self):
        pool = self._controller.self_play_db_conn_pool
        with pool.db_lock:
            cursor = pool.get_cursor()
            cursor.execute(
                """SELECT cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1""")
            row = cursor.fetchone()
            n = row[0]

            f = self._controller.training_params.window_size_function
            n = row[0]
            c = int(n - f(n))

            start = max(0, c)
            end = n

            self._master_list_slice.set_bounds(cursor, start, end)
            cursor.close()

    def _extend_master_list_from_fork(self, gen):
        start, end = self._controller.organizer.fork_info.train_windows[gen]

        pool = self._controller.self_play_db_conn_pool
        with pool.db_lock:
            cursor = pool.get_cursor()
            self._master_list_slice.set_bounds(cursor, start, end, gen)
            cursor.close()

    def _load_last_checkpoint(self):
        """
        If a prior checkpoint exists, does the following:

        - Sets self._net
        - Sets self._opt
        """
        organizer = self._controller.organizer
        checkpoint_info = organizer.get_latest_checkpoint_info()
        if checkpoint_info is None:
            return

        gen = checkpoint_info.generation
        checkpoint_filename = organizer.get_checkpoint_filename(gen)
        logger.info(f'Loading checkpoint: {checkpoint_filename}')

        # copying the checkpoint to somewhere local first seems to bypass some sort of
        # filesystem issue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.pt')
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
        self._net.cuda(device=self._controller.params.cuda_device)
        self._net.train()

        training_params = self._controller.training_params
        learning_rate = training_params.learning_rate
        momentum = training_params.momentum
        weight_decay = training_params.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

    def _get_net_and_optimizer(self) -> Tuple[Model, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        organizer = self._controller.organizer
        checkpoint_info = organizer.get_latest_checkpoint_info()
        if checkpoint_info is None:
            shape_info_dict = self._game_log_reader.shape_info_dict
            model_cfg = self._controller.params.model_cfg
            self._net = Model(self._controller.game_spec.model_configs[model_cfg](shape_info_dict))
            self._init_net_and_opt()
        else:
            self._load_last_checkpoint()

        return self._net, self._opt

    def _train_step_helper(self, dataset: PositionDataset, trainer: NetTrainer, gen: Generation):
        try:
            training_params = self._controller.training_params
            minibatch_size = training_params.minibatch_size

            loader = DataLoader(
                dataset,
                batch_size=minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)

            net, optimizer = self._get_net_and_optimizer()

            table: GpuContentionTable = self._controller.get_gpu_lock_table_for_training()
            logger.debug(f'Training table: {table}')
            table.acquire_lock(Domain.TRAINING)
            stats = trainer.do_training_epoch(loader, net, optimizer, dataset)

            if stats is None:
                # Happens in premature-shutdown case. No need to release training gpu lock since
                # the whole process is shutting down.
                return

            stats.dump(logger.info)
            logger.info(f'Gen {gen} training complete')
            trainer.dump_timing_stats(logger.info)

            self._save_model(gen, net)
            self._record_stats(gen, stats)
            self._controller.reset_self_play_locks()
            table.release_lock(Domain.TRAINING)
            self._controller.handle_new_model()
        except:
            logger.error('Unexpected error in train_step():', exc_info=True)
            self._controller.request_shutdown(1)

    def _record_stats(self, gen: Generation, stats: TrainingStats):
        training_params = self._controller.training_params
        n_minibatches = training_params.minibatches_per_epoch
        minibatch_size = training_params.minibatch_size

        window_start = stats.window_start
        window_end = stats.window_end
        n_samples = stats.n_samples
        start_ts = stats.start_ts
        end_ts = stats.end_ts

        window = construct_window(self._last_sample_window, window_start, window_end, n_samples)
        self._last_sample_window = window

        head_data = []
        for head_stats in stats.substats_list:
            head_name = head_stats.name
            loss = head_stats.loss()
            accuracy = head_stats.accuracy()
            loss_weight = head_stats.loss_weight
            head_data.append((gen, head_name, loss, loss_weight, accuracy))

        with self._controller.training_db_conn_pool.db_lock:
            conn = self._controller.training_db_conn_pool.get_connection()
            cursor = conn.cursor()

            cursor.execute("""INSERT OR REPLACE INTO training (gen, training_start_ts, training_end_ts,
                minibatch_size, n_minibatches, window_start, window_end, window_sample_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                           (gen, start_ts, end_ts, minibatch_size, n_minibatches,
                            window.start, window.end, window.sample_rate))

            cursor.executemany("""INSERT OR REPLACE INTO training_heads (gen, head_name, loss, loss_weight, accuracy)
                VALUES (?, ?, ?, ?, ?)""", head_data)

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
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        self._latest_gen = gen
        logger.info(f'Checkpoint saved: {checkpoint_filename}')
        logger.info(f'Model saved: {model_filename}')
