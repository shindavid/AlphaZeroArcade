"""
The self-play process continuously produces a chronological sequence of positions, M.

The training process, on each epoch, chooses a window W corresponding to a suffix of M, and samples
R positions uniformly from W. It then trains on those R positions.

If doing so would cause the positions of W to be sampled more than K*|W| times in expectation over
the entire course of training, then the training process waits for more data to be generated. This
mechanism causes the training process to spend most of its time waiting for new data. This is
necessary to avoid overfitting.

We use KataGo's methodology for choosing the size of W - a shifted/scaled version of the function
f(n) = n^0.75, where n = |M|.

This module provides a class, GamesDatasetGenerator, that tracks the master sequence M. This class
produces GamesDataset objects, which correspond to a window W of M.
"""
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from alphazero.data.metadata import SelfPlayMetadata
from util.infinite_sequence import InfiniteSequence
from util.torch_util import Shape


class GamesDataset(Dataset):
    """
    A GamesDataset represents a slice of a master list of game-positions.
    """
    def __init__(self, self_play_metadata: SelfPlayMetadata, start: int, end: int):
        """
        self_play_metadata: effectively represents a master list (M) of positions

        Constructs a GamesDataset corresponding to M[start:end]
        """
        self.self_play_metadata = self_play_metadata
        self.n_total_positions = self_play_metadata.n_total_positions
        self.start = start
        self.end = end
        assert 0 <= start < end <= self.n_total_positions, (start, end)
        self.window = list(self.self_play_metadata.get_window(start, end))
        assert len(self.window) == end - start, (len(self.window), start, end)
        self.key_order: List[str] = []

    def __str__(self):
        first_gen = self.window[0].game_metadata.generation
        last_gen = self.window[-1].game_metadata.generation
        return f'GamesDataset(start={self.start}[gen-{first_gen}], end={self.end}[gen-{last_gen}])'

    def get_input_shape(self) -> Shape:
        """
        Peeks into the dataset to determine the shape of the input.

        Without this, we would need to hard-code the input shape into the model configuration,
        which could be cumbersome if we want to experiment with different ways of representing the
        input. For example, we may want to vary the number of previous positions of history we
        include in the input, or we may want to add some additional input feature planes.
        """
        position_metadata = self.window[0]
        game_metadata = position_metadata.game_metadata
        data = torch.jit.load(game_metadata.filename).state_dict()
        return data['input'].shape[1:]

    def get_target_names(self) -> List[str]:
        """
        Peeks into the dataset to find the names of all the targets.
        """
        position_metadata = self.window[0]
        game_metadata = position_metadata.game_metadata
        data = torch.jit.load(game_metadata.filename).state_dict()
        names = list(data.keys())
        return [n for n in names if n != 'input']

    def set_key_order(self, target_names: List[str]):
        """
        The key order determines the order in which the data is returned by __getitem__.

        This must be called prior to iterating over the dataset.
        """
        self.key_order = ['input'] + target_names

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        assert self.key_order, 'Must call set_key_order() before iterating over GamesDataset'
        assert 0 <= idx < len(self), (idx, self.start, self.end)
        position_metadata = self.window[idx]
        game_metadata = position_metadata.game_metadata
        p = position_metadata.position_index
        try:
            data = torch.jit.load(game_metadata.filename).state_dict()
        except:
            raise Exception('Could not load data from file: {}'.format(game_metadata.filename))
        return [data[key][p] for key in self.key_order]


class GamesDatasetGenerator:
    """
    Example usage:

    sample_limit = 10
    generator = GamesDatasetGenerator(self_play_data_dir, sample_limit)
    while True:
        dataset = generator.get_next_dataset(loader_size)
        if dataset is None:
            # wait for more data to be generated
            time.sleep(5)
            continue

        ...  # do an epoch network training here by iterating over dataset

        generator.record_dataset_usage(dataset, num_positions_sampled)

    In the above, self_play_data_dir is a directory that contains a master list of positions. A
    GamesDatasetGenerator effectively maintains a list M that is initially equal to this master
    list.

    On every loop iteration, the generator extends M with any newly generated data, and returns a
    GamesDataset corresponding to a subsequence of M.

    When record_dataset_usage() is called, the generator marks that num_positions_sampled positions
    were uniformly randomly sampled from dataset.
    """
    def __init__(self, self_play_data_dir: str, sample_limit: Optional[int] = None):
        """
        sample_limit dictates the maximum expected number of times any given training row can be
        used by the training process. If None, then there is no limit.

        KataGo uses a limit of 4. LeelaChessZero effectively uses a limit of 8.
        """
        self.expected_sample_counts = InfiniteSequence()
        self.self_play_metadata = SelfPlayMetadata(self_play_data_dir)
        self.self_play_data_dir = self_play_data_dir
        self.sample_limit = sample_limit

    @property
    def n_total_positions(self) -> int:
        return self.self_play_metadata.n_total_positions

    def init_to_sample_limit(self):
        self.expected_sample_counts[:self.n_total_positions] = self.sample_limit

    def get_next_dataset(self, loader_size: int, verbose: bool = False) -> Optional[GamesDataset]:
        """
        Returns a DataLoader corresponding to a slice of the master list of size loader_size. If
        use_prefix is True, the slice is taken from the beginning of the master list. Otherwise, it
        is taken from the end.

        If there are not enough positions available, returns None. If it returns None, it is
        recommended to sleep for a few seconds before trying again, to avoid thrashing the
        filesystem.
        """
        self.self_play_metadata.refresh()
        n_total_positions = self.self_play_metadata.n_total_positions
        end = n_total_positions
        start = end - loader_size

        sample_sum = self.expected_sample_counts[start:end].sum()
        if verbose:
            print('Total positions:', n_total_positions)
            print('expected_sampled_counts:\n ',
                  self.expected_sample_counts.to_string(delim='\n  ', cap=self.sample_limit))
            print('sample_sum: %.3f' % sample_sum)
            print('loader_size: %d' % loader_size)

            start_pct = 100. * start / n_total_positions
            end_pct = 100. * end / n_total_positions
            print('Data range: [%.3f%% - %.3f%%]' % (start_pct, end_pct))

        if sample_sum > self.sample_limit * loader_size:
            return None

        return GamesDataset(self.self_play_metadata, start, end)

    def record_dataset_usage(self, dataset: GamesDataset, num_positions_sampled: int):
        """
        Marks that num_positions_sampled positions were uniformly randomly sampled from dataset.
        It then identifies any positions in M that were sampled >= sampled_limit times, and
        effectively discards those positions and all positions preceding them from M.
        """
        start = dataset.start
        end = dataset.end
        x = num_positions_sampled / (end - start)
        self.expected_sample_counts[start:end] += x


def get_katago_sample_size(n_total: int, alpha=0.75, beta=0.4, c=250000) -> int:
    """
    Returns the number of positions from which to sample, as a function of the total number of
    positions in the master list.

    This is the sublinear curve f(n) = n^alpha but rescaled so that f(c) = c and f'(c) = beta.

    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))
