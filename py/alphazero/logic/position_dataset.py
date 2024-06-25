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
from alphazero.logic.game_log_reader import GameLogReader
from shared.net_modules import ShapeInfo

import numpy as np
import os
import sqlite3
from torch.utils.data import Dataset

from typing import List, Optional


pos_dtype = np.dtype([('client_id', 'i4'),
                      ('gen', 'i4'),
                      ('game_end_ts', 'i8'),
                      ('pos_index', 'i4')])


class PositionListSlice:
    def __init__(self):
        self._positions = np.zeros(0, dtype=pos_dtype)
        self._start_index = 0  # index of master list, corresponds to self._positions[0]
        self._end_index = 0  # index of master list
        self._last_game_id = -1
        self._max_forked_client_id = -1

    def set_max_forked_client_id(self, max_forked_client_id: int):
        self._max_forked_client_id = max_forked_client_id

    def __len__(self):
        return self._end_index - self._start_index

    def __getitem__(self, idx):
        return self._positions[idx]

    @property
    def start_index(self):
        return self._start_index

    @property
    def end_index(self):
        return self._end_index

    @property
    def max_forked_client_id(self):
        return self._max_forked_client_id

    def _get_positions(self, c: sqlite3.Cursor, gen: Optional[int]):
        if gen is None:
            c.execute("""SELECT id, client_id, gen, end_timestamp, augmented_positions
                        FROM games WHERE id > ?""", (self._last_game_id,))
        else:
            c.execute("""SELECT id, client_id, gen, end_timestamp, augmented_positions
                    FROM games WHERE id > ? AND gen < ?""", (self._last_game_id, gen))
        for row in c:
            game_id, client_id, gen, end_timestamp, augmented_positions = row
            assert game_id >= self._last_game_id, (game_id, self._last_game_id)
            self._last_game_id = game_id
            for pos_index in range(augmented_positions):
                yield (client_id, gen, end_timestamp, pos_index)

    def set_bounds(self, cursor: sqlite3.Cursor, start: int, end: int, gen: Optional[int]=None):
        assert start >= self._start_index, (start, end, self._start_index)
        assert 0 <= start < end, (start, end, self._start_index)

        positions = np.fromiter(self._get_positions(cursor, gen), dtype=pos_dtype)
        self._positions = np.concatenate([self._positions, positions])

        n_rows_to_cut = start - self._start_index
        self._positions = self._positions[n_rows_to_cut:]
        self._start_index = start
        self._end_index = end


class PositionDataset(Dataset):
    def __init__(self, base_dir: str, forked_base_dir: Optional[str], positions: PositionListSlice,
                 game_log_reader: GameLogReader):
        self._base_dir = base_dir
        self._forked_base_dir = forked_base_dir
        self._positions = positions
        self._game_log_reader = game_log_reader
        self._input_shape_info: ShapeInfo = game_log_reader.shape_info_dict['input']
        self._target_shape_infos: List[ShapeInfo] = []

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

    def set_key_order(self, target_names: List[str]):
        """
        The key order determines the order in which the data is returned by __getitem__.

        This must be called prior to iterating over the dataset.
        """
        shape_info_dict = self._game_log_reader.shape_info_dict
        self._target_shape_infos = [shape_info_dict[name] for name in target_names]

    def _get_filename(self, client_id: int, gen: int, end_ts: int) -> str:
        use_forked_dir = client_id <= self._positions.max_forked_client_id
        base_dir = self._forked_base_dir if use_forked_dir else self._base_dir
        return os.path.join(base_dir, 'self-play-data', f'client-{client_id}',
                            f'gen-{gen}', f'{end_ts}.log')

    def __len__(self):
        return len(self._positions)

    def __getitem__(self, idx):
        client_id, gen, end_timestamp, pos_index = self._positions[idx]
        filename = self._get_filename(client_id, gen, end_timestamp)

        try:
            log = self._game_log_reader.open_log(filename)
            output = self._game_log_reader.create_tensors(log, self._input_shape_info,
                                                          self._target_shape_infos, pos_index)
            self._game_log_reader.close_log(log)
            return output
        except:
            raise Exception(f'Could not load data from file: {filename}')
