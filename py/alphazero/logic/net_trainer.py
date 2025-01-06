from typing import List
import torch
from torch import optim

from alphazero.logic.custom_types import Generation
from alphazero.logic.position_dataset import PositionDataset
from shared.net_modules import Head, Model
from util.logging_util import get_logger
from util.torch_util import apply_mask


import time
from typing import Optional


logger = get_logger()


class EvaluationResults:
    def __init__(self, labels, outputs, loss):
        self.labels = labels
        self.outputs = outputs
        self.loss = loss

    def __len__(self):
        return len(self.labels)


class TrainingSubStats:
    max_descr_len = 0

    def __init__(self, head: Head, loss_weight: float):
        self.head = head
        self.loss_weight = loss_weight
        self.loss_num = 0.0
        self.den = 0

        TrainingSubStats.max_descr_len = max(TrainingSubStats.max_descr_len, len(self.descr))

    @property
    def name(self) -> str:
        return self.head.name

    @property
    def descr(self) -> str:
        return self.head.name

    def update(self, results: EvaluationResults):
        n = len(results)
        self.loss_num += float(results.loss.item()) * n
        self.den += n

    def loss(self):
        return self.loss_num / self.den if self.den else 0.0

    def dump(self, total_loss, print_fn=print):
        output = [self.descr.rjust(TrainingSubStats.max_descr_len)]

        loss = self.loss()
        weight = self.loss_weight
        loss_pct = 100. * loss * weight / total_loss if total_loss else 0.0
        output.append(' loss: %8.6f * %6.3f = %8.6f [%6.3f%%]' % (
            loss, weight, loss * weight, loss_pct))

        print_fn(''.join(output))

    def to_json(self):
        return {
            'loss': self.loss(),
            'loss_weight': self.loss_weight,
            }

    @staticmethod
    def dump_total_loss(total_loss, print_fn=print):
        output = ['total'.rjust(TrainingSubStats.max_descr_len)]
        output.append(' loss:                   = %8.6f' % total_loss)
        print_fn(''.join(output))


class TrainingStats:
    def __init__(self, gen: Generation, minibatch_size: int, window_start: int,
                 window_end: int, net: Model):
        self.gen = gen
        self.minibatch_size = minibatch_size
        self.start_ts = 0
        self.end_ts = 0
        self.window_start = window_start
        self.window_end = window_end
        self.window_sample_rate = 0.0

        self.n_minibatches_processed = 0
        self.n_samples = 0
        self.substats_list = [TrainingSubStats(head, net.loss_weights[head.name])
                              for head in net.heads]

    def update(self, results_list: List[EvaluationResults], n_samples):
        self.n_samples += n_samples
        for results, substats in zip(results_list, self.substats_list):
            substats.update(results)

    def dump(self, print_fn=print):
        total_loss = 0
        for substats in self.substats_list:
            total_loss += substats.loss() * substats.loss_weight

        for substats in self.substats_list:
            substats.dump(total_loss, print_fn=print_fn)

        TrainingSubStats.dump_total_loss(total_loss, print_fn=print_fn)

    def to_json(self):
        substats = { s.name: s.to_json() for s in self.substats_list }
        return {
            'gen': self.gen,
            'start_ts': self.start_ts,
            'end_ts': self.end_ts,
            'window_start': self.window_start,
            'window_end': self.window_end,
            'window_sample_rate': self.window_sample_rate,
            'n_samples': self.n_samples,
            'minibatch_size': self.minibatch_size,
            'n_minibatches': self.n_minibatches_processed,
            'substats': substats,
        }


class NetTrainer:
    def __init__(self, gen: Generation, n_minibatches_to_process: int=-1,
                 py_cuda_device_str: str='cuda:0'):
        self._shutdown_in_progress = False
        self.gen = gen
        self.n_minibatches_to_process = n_minibatches_to_process
        self.py_cuda_device_str = py_cuda_device_str
        self.reset()

    def reset(self):
        self.for_loop_time = 0
        self.t0 = time.time()

    def shutdown(self):
        self._shutdown_in_progress = True

    def do_training_epoch(self,
                          loader: torch.utils.data.DataLoader,
                          net: Model,
                          optimizer: optim.Optimizer,
                          dataset: PositionDataset) -> Optional[TrainingStats]:
        """
        Performs a training epoch by processing data from loader. Stops when either
        self.n_minibatches_to_process minibatch updates have been performed or until all the data in
        loader has been processed, whichever comes first. If self.n_minibatches_to_process is
        negative, that is treated like infinity.

        If a separate thread calls self.shutdown(), then this exits prematurely and returns None
        """
        dataset.set_key_order(net.target_names)

        loss_fns = [head.target.loss_fn() for head in net.heads]
        loss_weights = [net.loss_weights[head.name] for head in net.heads]

        window_start = dataset.start_index
        window_end = dataset.end_index

        first = True

        start_ts = time.time_ns()
        n_samples = 0
        stats = TrainingStats(self.gen, loader.batch_size, window_start, window_end, net)
        for data in loader:
            if self._shutdown_in_progress:
                return None
            t1 = time.time()
            d = len(data)
            assert d % 2 == 1, d
            dd = (d + 1) // 2
            inputs = data[0]
            labels_list = data[1:dd]
            masks = data[dd:]

            inputs = inputs.type(torch.float32).to(self.py_cuda_device_str)

            labels_list = [labels.to(self.py_cuda_device_str) for labels in labels_list]
            masks_list = [mask.to(self.py_cuda_device_str).squeeze(-1) for mask in masks]

            optimizer.zero_grad()
            outputs_list = net(inputs)
            assert len(outputs_list) == len(labels_list)

            labels_list = [apply_mask(y_hat, mask) for mask, y_hat in zip(masks_list, labels_list)]
            outputs_list = [apply_mask(y, mask) for mask, y in zip(masks_list, outputs_list)]

            loss_list = [loss_fn(outputs, labels) for loss_fn, outputs, labels in
                            zip(loss_fns, outputs_list, labels_list)]

            loss = sum([loss * loss_weight for loss, loss_weight in zip(loss_list, loss_weights)])

            results_list = [EvaluationResults(labels, outputs, subloss) for labels, outputs, subloss in
                            zip(labels_list, outputs_list, loss_list)]

            n_samples += len(inputs)
            stats.update(results_list, len(inputs))

            loss.backward()
            optimizer.step()
            stats.n_minibatches_processed += 1
            t2 = time.time()
            self.for_loop_time += t2 - t1
            if stats.n_minibatches_processed == self.n_minibatches_to_process:
                break
            if self._shutdown_in_progress:
                return None

        end_ts = time.time_ns()

        window_sample_rate = n_samples / (window_end - window_start)

        stats.start_ts = start_ts
        stats.end_ts = end_ts
        stats.window_sample_rate = window_sample_rate

        return stats

    def dump_timing_stats(self, print_fn=print):
        total_time = time.time() - self.t0
        data_loading_time = total_time - self.for_loop_time
        print_fn(f'Data loading time: {data_loading_time:10.3f} seconds')
        print_fn(f'Training time:     {self.for_loop_time:10.3f} seconds')
