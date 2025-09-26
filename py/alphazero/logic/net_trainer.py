from typing import List
from torch import optim

from alphazero.logic.custom_types import Generation
from alphazero.logic.game_log_reader import GameLogReader
from shared.net_modules import Head, Model
from util.torch_util import apply_mask

import logging
import time
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class EvaluationResults:
    def __init__(self, labels, outputs, loss):
        self.labels = labels
        self.outputs = outputs
        self.loss = loss

    def __len__(self):
        return len(self.labels)


class TrainingSubStats:
    max_descr_len = 0

    def __init__(self, name: str, loss_weight: float):
        self.name = name
        self.loss_weight = loss_weight
        self.loss_num = 0.0
        self.den = 0

        TrainingSubStats.max_descr_len = max(TrainingSubStats.max_descr_len, len(self.descr))

    @property
    def descr(self) -> str:
        return self.name

    def update(self, results: EvaluationResults):
        n = len(results)
        self.loss_num += float(results.loss.item()) * n
        self.den += n

    def loss(self):
        return self.loss_num / self.den if self.den else 0.0

    def dump(self, total_loss, log_level):
        output = [self.descr.rjust(TrainingSubStats.max_descr_len)]

        loss = self.loss()
        weight = self.loss_weight
        loss_pct = 100. * loss * weight / total_loss if total_loss else 0.0
        output.append(' loss: %8.6f * %6.3f = %8.6f [%6.3f%%]' % (
            loss, weight, loss * weight, loss_pct))

        logger.log(log_level, ''.join(output))

    @staticmethod
    def dump_total_loss(total_loss, log_level):
        output = ['total'.rjust(TrainingSubStats.max_descr_len)]
        output.append(' loss:                   = %8.6f' % total_loss)
        logger.log(log_level, ''.join(output))


class TrainingStats:
    def __init__(self, gen: Generation, minibatch_size: int, window_start: int,
                 window_end: int, net: Model, loss_weights: Dict[str, float]):
        self.gen = gen
        self.minibatch_size = minibatch_size
        self.window_start = window_start
        self.window_end = window_end
        self.window_sample_rate = 0.0

        self.n_minibatches_processed = 0
        self.n_samples = 0
        self.substats_list = [TrainingSubStats(name, loss_weights[name])
                              for name in net.target_names]

    def update(self, results_list: List[EvaluationResults], n_samples):
        self.n_samples += n_samples
        for results, substats in zip(results_list, self.substats_list):
            substats.update(results)

    def dump(self, log_level):
        if logger.level > log_level:
            return

        total_loss = 0
        for substats in self.substats_list:
            total_loss += substats.loss() * substats.loss_weight

        for substats in self.substats_list:
            substats.dump(total_loss, log_level)

        TrainingSubStats.dump_total_loss(total_loss, log_level)


class NetTrainer:
    def __init__(self, gen: Generation, n_minibatches_to_process: int=-1,
                 py_cuda_device_str: str='cuda:0'):
        self._shutdown_in_progress = False
        self.gen = gen
        self.n_minibatches_to_process = n_minibatches_to_process
        self.py_cuda_device_str = py_cuda_device_str

    def shutdown(self):
        self._shutdown_in_progress = True

    def do_training_epoch(self,
                          reader: GameLogReader,
                          net: Model,
                          optimizer: optim.Optimizer,
                          minibatch_size: int,
                          n_minibatches: int,
                          window_start: int,
                          window_end: int,
                          gen: Generation,
                          loss_weights: Dict[str, float]) -> Optional[TrainingStats]:
        """
        Performs a training epoch by processing data from loader. Stops when either
        self.n_minibatches_to_process minibatch updates have been performed or until all the data in
        loader has been processed, whichever comes first. If self.n_minibatches_to_process is
        negative, that is treated like infinity.

        If a separate thread calls self.shutdown(), then this exits prematurely and returns None
        """
        t0 = time.time()
        train_time = 0.0

        data_batches = reader.create_data_batches(
            minibatch_size, n_minibatches, window_start, window_end, net._target_names, gen)

        loss_fns = [head.target.loss_fn() for head in net.heads]
        loss_weights_list = [loss_weights[name] for name in net.target_names]

        n_samples = 0
        stats = TrainingStats(self.gen, minibatch_size, window_start, window_end, net, loss_weights)
        for batch in data_batches:
            if self._shutdown_in_progress:
                return None

            t1 = time.time()
            inputs = batch.input_tensor
            labels = batch.target_tensors
            masks = batch.target_masks

            optimizer.zero_grad()
            outputs = net(inputs)
            assert len(outputs) == len(labels)

            labels = [apply_mask(y_hat, mask) for mask, y_hat in zip(masks, labels)]
            outputs = [apply_mask(y, mask) for mask, y in zip(masks, outputs)]
            losses = [f(y_hat, y) for f, y_hat, y in zip(loss_fns, outputs, labels)]
            loss = sum([l * w for l, w in zip(losses, loss_weights_list)])
            results_list = [EvaluationResults(*x) for x in zip(labels, outputs, losses)]

            n_samples += len(inputs)
            stats.update(results_list, len(inputs))

            loss.backward()
            optimizer.step()

            t2 = time.time()
            train_time += t2 - t1
            stats.n_minibatches_processed += 1
            if stats.n_minibatches_processed == self.n_minibatches_to_process:
                break
            if self._shutdown_in_progress:
                return None

        window_sample_rate = n_samples / (window_end - window_start)

        stats.window_sample_rate = window_sample_rate
        t3 = time.time()

        total_time = t3 - t0
        load_time = total_time - train_time

        stats.dump(logging.INFO)
        logger.info('Data loading time:   %10.3f seconds', load_time)
        logger.info('Training time:       %10.3f seconds', train_time)

        return stats
