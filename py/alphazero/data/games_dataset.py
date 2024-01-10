"""
The self-play process continuously appends to a master sequence of positions, M.

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

import numpy as np
import os
import sqlite3
import torch
from torch.utils.data import Dataset

from alphazero.sample_window_logic import SamplingParams, Window, get_required_dataset_size
from util.torch_util import Shape


pos_dtype = np.dtype([('client_id', 'i4'),
                      ('gen', 'i4'),
                      ('game_end_ts', 'i8'),
                      ('pos_index', 'i4')])


class PositionListSlice:
    def __init__(self):
        self._positions = np.zeros(0, dtype=pos_dtype)
        self._start_index = 0  # index of master list
        self._last_game_id = -1

    def __len__(self):
        return len(self._positions)

    def __getitem__(self, idx):
        return self._positions[idx]

    @property
    def start_index(self):
        """
        Returns the index of the first position in the master list.
        """
        return self._start_index

    @property
    def end_index(self):
        """
        Returns one plus the index of the last position in the master list.
        """
        return self._start_index + len(self._positions)

    def _get_positions(self, c: sqlite3.Cursor):
        c.execute("""SELECT id, client_id, gen, end_timestamp, augmented_positions
                  FROM games WHERE id > ?""", (self._last_game_id,))
        for row in c:
            game_id, client_id, gen, end_timestamp, augmented_positions = row
            assert game_id >= self._last_game_id, (game_id, self._last_game_id)
            self._last_game_id = game_id
            for pos_index in range(augmented_positions):
                yield (client_id, gen, end_timestamp, pos_index)

    def extend(self, cursor: sqlite3.Cursor):
        """
        Extends the list of positions with newly added games from the database.
        """
        positions = np.fromiter(self._get_positions(cursor), dtype=pos_dtype)
        self._positions = np.concatenate([self._positions, positions])

    def set_start_index(self, start_index: int):
        n_rows_to_cut = start_index - self._start_index
        positions = self._positions[n_rows_to_cut:]
        assert start_index >= self._start_index, (start_index, self._start_index)
        assert len(positions) > 0, (start_index, self._start_index)

        self._positions = positions
        self._start_index = start_index


class PositionDataset(Dataset):
    def __init__(self, base_dir: str, positions: PositionListSlice):
        self._base_dir = base_dir
        self._positions = positions
        self._key_order: List[str] = []

    def announce_sampling(self, print_func):
        dataset_size = len(self)
        n_total_positions = self._positions.end_index
        first_gen = self._positions[0]['gen']
        last_gen = self._positions[-1]['gen']
        assert first_gen <= last_gen
        if first_gen == last_gen:
            gen_str = f'gen {first_gen}'
        else:
            gen_str = f'gens {first_gen} to {last_gen}'
        print_func(
            'Sampling from %s of %s (%.1f%%) positions (%s)' %
            (dataset_size, n_total_positions, 100. * dataset_size / n_total_positions, gen_str))

    @property
    def start_index(self):
        """
        Returns the index of the first position in the master list.
        """
        return self._positions.start_index

    @property
    def end_index(self):
        """
        Returns one plus the index of the last position in the master list.
        """
        return self._positions.end_index

    def _get_data_and_pos_index(self, idx):
        client_id, gen, end_timestamp, pos_index = self._positions[idx]
        filename = self._get_filename(client_id, gen, end_timestamp)

        try:
            data = torch.jit.load(filename).state_dict()
        except:
            raise Exception(f'Could not load data from file: {filename}')
        return data, pos_index

    def get_input_shape(self) -> Shape:
        """
        Peeks into the dataset to determine the shape of the input.

        Without this, we would need to hard-code the input shape into the model configuration,
        which could be cumbersome if we want to experiment with different ways of representing the
        input. For example, we may want to vary the number of previous positions of history we
        include in the input, or we may want to add some additional input feature planes.
        """
        data, pos_index = self._get_data_and_pos_index(0)
        return data['input'][pos_index].shape

    def get_target_names(self) -> List[str]:
        """
        Peeks into the dataset to find the names of all the targets.
        """
        data, _ = self._get_data_and_pos_index(0)
        names = list(data.keys())
        return [n for n in names if n != 'input']

    def set_key_order(self, target_names: List[str]):
        """
        The key order determines the order in which the data is returned by __getitem__.

        This must be called prior to iterating over the dataset.
        """
        self._key_order = ['input'] + target_names

    def _get_filename(self, client_id: int, gen: int, end_ts: int) -> str:
        return os.path.join(self._base_dir, 'self-play-data', f'client-{client_id}',
                            f'gen-{gen}', f'{end_ts}.ptd')

    def __len__(self):
        return len(self._positions)

    def __getitem__(self, idx):
        data, pos_index = self._get_data_and_pos_index(idx)
        return [data[key][pos_index] for key in self._key_order]


# class PositionDatasetGenerator:
#     def __init__(self, base_dir: str, db_conn: sqlite3.Connection, params: SamplingParams):
#         self._base_dir = base_dir
#         self._master_list = PositionListSlice()
#         self._params = params
#         self._db_conn = db_conn

#     @property
#     def master_length(self):
#         return self._master_list.end_index

#     def get_next_dataset(self, last_sample_window: Window) -> Optional[PositionDataset]:
#         """
#         Returns the next dataset for sampling. Returns None if there are not enough positions.
#         """
#         cursor = self._db_conn.cursor()
#         self._master_list.extend(cursor)

#         n = self.master_length
#         f = get_required_dataset_size(self._params, last_sample_window)
#         if n < f:
#             return None

#         c = n - f
#         self._master_list.set_start_index(c)
#         assert len(self._master_list) == f, (len(self._master_list), f)
#         return PositionDataset(self._base_dir, self._master_list)
