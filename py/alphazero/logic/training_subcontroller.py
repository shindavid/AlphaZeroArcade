from alphazero.data.position_dataset import PositionDataset, PositionListSlice
from alphazero.logic.aux_subcontroller import AuxSubcontroller
from alphazero.logic.custom_types import ChildThreadError, Generation
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.loop_control_data import LoopControlData
from alphazero.logic.net_trainer import NetTrainer, TrainingStats
from alphazero.logic.sample_window_logic import Window, construct_window, get_required_dataset_size
from net_modules import Model
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


class TrainingSubcontroller:
    """
    Used by the LoopController to manage training.
    """

    def __init__(self, aux_controller: AuxSubcontroller):
        self.aux_controller = aux_controller

        self._net = None
        self._opt = None
        self._cuda_device = self.data.params.cuda_device

        self._ready_event = threading.Event()
        self._lock = threading.Lock()

        # These members are initialized lazily, so that we don't need to read the database until we
        # actually need the data.
        self._last_sample_window: Optional[Window] = None
        self._master_list_length: Optional[int] = None
        self._master_list_length_for_next_train_loop: Optional[int] = None
        self._master_list_slice = PositionListSlice()

    @property
    def data(self) -> LoopControlData:
        return self.aux_controller.data

    @property
    def organizer(self) -> DirectoryOrganizer:
        return self.data.organizer

    def setup(self):
        self._last_sample_window = self._load_last_sample_window()

        # The length of the master_list can be computed on-demand by reading the database. To
        # avoid doing this repeatedly, we grab the value once at start-up, store it as a member, and
        # then update it manually whenever we add new games to the database.
        self._master_list_length = self._fetch_num_total_augmented_positions()

        # This is the length that the master_list needs to be before we can start a new train loop.
        # Initialized lazily.
        self._master_list_length_for_next_train_loop = 0

    def _fetch_num_total_augmented_positions(self) -> int:
        with self.data.self_play_db_conn_pool.db_lock:
            # Return cumulative_augmented_positions for the last row of games:
            cursor = self.data.self_play_db_conn_pool.get_cursor()
            cursor.execute("""SELECT cumulative_augmented_positions FROM games
                           ORDER BY id DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return 0
        return row[0]

    def _load_last_sample_window(self) -> Window:
        with self.data.training_db_conn_pool.db_lock:
            cursor = self.data.training_db_conn_pool.get_cursor()
            cursor.execute("""SELECT window_start, window_end, window_sample_rate
                            FROM training ORDER BY gen DESC LIMIT 1""")
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            # kZero-style initialization of sample window
            samples_per_window = self.data.training_params.samples_per_window()
            target_sample_rate = self.data.training_params.target_sample_rate
            return Window(0, samples_per_window, target_sample_rate)
        return Window(*row)

    def increment_master_list_length(self, n: int):
        with self._lock:
            self._master_list_length += n
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                self._ready_event.set()

    def wait_until_enough_training_data(self):
        with self._lock:
            self._master_list_length_for_next_train_loop = get_required_dataset_size(
                self.data.training_params, self._last_sample_window)
            if self._master_list_length >= self._master_list_length_for_next_train_loop:
                return
            self._ready_event.clear()

        # TODO: progress-bar (use module tqdm)
        logger.info('Waiting for more training data...')
        self._ready_event.wait()

    def extend_master_list(self):
        with self.data.self_play_db_conn_pool.db_lock:
            cursor = self.data.self_play_db_conn_pool.get_cursor()
            cursor.execute(
                """SELECT cumulative_augmented_positions FROM games ORDER BY id DESC LIMIT 1""")
            row = cursor.fetchone()
            n = row[0]

            f = self.data.training_params.window_size_function
            n = row[0]
            c = int(n - f(n))

            start = c
            end = n

            self._master_list_slice.set_bounds(cursor, start, end)
            cursor.close()

    def load_last_checkpoint(self):
        """
        If a prior checkpoint exists, does the following:

        - Sets self._net
        - Sets self._opt
        """
        checkpoint_info = self.organizer.get_latest_checkpoint_info()
        if checkpoint_info is None:
            return

        gen = checkpoint_info.generation
        checkpoint_filename = self.organizer.get_checkpoint_filename(gen)
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
        self._net.cuda(device=self._cuda_device)
        self._net.train()

        training_params = self.data.training_params
        learning_rate = training_params.learning_rate
        momentum = training_params.momentum
        weight_decay = training_params.weight_decay
        self._opt = optim.SGD(self._net.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

    def get_net_and_optimizer(self, loader: DataLoader) -> Tuple[Model, optim.Optimizer]:
        if self._net is not None:
            return self._net, self._opt

        checkpoint_info = self.organizer.get_latest_checkpoint_info()
        if checkpoint_info is None:
            dataset: PositionDataset = loader.dataset
            input_shape = dataset.get_input_shape()
            target_names = dataset.get_target_names()
            self._net = Model(
                self.data.game_spec.model_configs[self.data.params.model_cfg](input_shape))
            self._net.validate_targets(target_names)
            self._init_net_and_opt()
            logger.info(f'Creating new net with input shape {input_shape}')
        else:
            self.load_last_checkpoint()

        return self._net, self._opt

    def train_gen1_model_if_necessary(self):
        gen = 1
        model_filename = self.organizer.get_model_filename(gen)
        if os.path.isfile(model_filename):
            return

        self.train_step()
        if self.data.error_signaled():
            raise ChildThreadError('Error signaled during train_step')

    def train_step(self):
        """
        Performs a train step.

        Uses a separate thread to ensure that the DataLoader is properly cleaned up after the
        train step is complete.
        """
        thread = threading.Thread(target=self._train_step_helper, name='train_step', daemon=True)
        thread.start()
        thread.join()

    def _train_step_helper(self):
        try:
            gen = self.data.organizer.get_latest_model_generation() + 1
            self.extend_master_list()

            dataset = PositionDataset(self.data.organizer.base_dir, self._master_list_slice)

            logger.info('******************************')
            logger.info(f'Train gen:{gen}')
            dataset.announce_sampling(logger.info)

            n_minibatches = self.data.training_params.minibatches_per_epoch
            minibatch_size = self.data.training_params.minibatch_size

            trainer = NetTrainer(gen, n_minibatches, self.data.params.cuda_device)

            loader = DataLoader(
                dataset,
                batch_size=minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)

            net, optimizer = self.get_net_and_optimizer(loader)

            self.aux_controller.pause_shared_gpu_self_play_clients()

            stats = trainer.do_training_epoch(loader, net, optimizer, dataset)
            stats.dump(logger.info)
            logger.info(f'Gen {gen} training complete')
            trainer.dump_timing_stats(logger.info)

            self._save_model(gen, net)
            self._record_stats(gen, stats)
            self.data.close_db_conns(threading.get_ident())
            self.aux_controller.reload_weights(gen)
            self.aux_controller.broadcast_new_model(gen)
        except:
            logger.error('Unexpected error in train_step():', exc_info=True)
            self.data.signal_error()

    def _record_stats(self, gen: Generation, stats: TrainingStats):
        n_minibatches = self.data.training_params.minibatches_per_epoch
        minibatch_size = self.data.training_params.minibatch_size

        window_start = stats.window_start
        window_end = stats.window_end
        n_samples = stats.n_samples
        start_ts = stats.start_ts
        end_ts = stats.end_ts

        window = construct_window(
            self._last_sample_window, window_start, window_end, n_samples)
        self._last_sample_window = window

        head_data = []
        for head_stats in stats.substats_list:
            head_name = head_stats.name
            loss = head_stats.loss()
            accuracy = head_stats.accuracy()
            loss_weight = head_stats.loss_weight
            head_data.append((gen, head_name, loss, loss_weight, accuracy))

        with self.data.training_db_conn_pool.db_lock:
            conn = self.data.training_db_conn_pool.get_connection()
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
        checkpoint_filename = self.organizer.get_checkpoint_filename(gen)
        model_filename = self.organizer.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        checkpoint = {}
        net.add_to_checkpoint(checkpoint)
        torch.save(checkpoint, tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        logger.info(f'Checkpoint saved: {checkpoint_filename}')
        logger.info(f'Model saved: {model_filename}')
