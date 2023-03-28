import torch
from typing import List
import random
from util.torch_util import Shape
from alphazero.optimization_args import ModelingArgs
from alphazero.data.metadata import (
    SelfPlayMetadata,
    SelfPlayPositionMetadata,
)

class DataLoader:
    def __init__(self, self_play_data_dir: str):
        self.self_play_metadata = SelfPlayMetadata(self_play_data_dir)
        self.n_total_games = self.self_play_metadata.n_total_games
        self.n_total_positions = self.self_play_metadata.n_total_positions
        self.n_window = compute_n_window(self.n_total_positions)
        self.window = self.self_play_metadata.get_window(self.n_window)

        self._returned_snapshots = 0
        self._index = len(self.window)

    def get_input_shape(self) -> Shape:
        for position_metadata in self.window:
            game_metadata = position_metadata.game_metadata
            data = torch.jit.load(game_metadata.filename).state_dict()
            return data['input'].shape[1:]
        raise Exception('Could not determine input shape!')

    def __iter__(self):
        return self

    def __next__(self):
        if self._returned_snapshots == ModelingArgs.snapshot_steps:
            raise StopIteration

        self._returned_snapshots += 1
        minibatch: List[SelfPlayPositionMetadata] = []
        for _ in range(ModelingArgs.minibatch_size):
            self._add_to_minibatch(minibatch)

        input_data = []
        policy_data = []
        value_data = []

        for position_metadata in minibatch:
            game_metadata = position_metadata.game_metadata
            p = position_metadata.position_index
            data = torch.jit.load(game_metadata.filename).state_dict()
            input_data.append(data['input'][p:p+1])
            policy_data.append(data['policy'][p:p+1])
            value_data.append(data['value'][p:p+1])

        input_data = torch.concat(input_data)
        policy_data = torch.concat(policy_data)
        value_data = torch.concat(value_data)

        return input_data, value_data, policy_data

    def _add_to_minibatch(self, minibatch: List[SelfPlayPositionMetadata]):
        if self._index == len(self.window):
            random.shuffle(self.window)
            self._index = 0

        position_metadata = self.window[self._index]
        minibatch.append(position_metadata)
        self._index += 1


def compute_n_window(n_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))