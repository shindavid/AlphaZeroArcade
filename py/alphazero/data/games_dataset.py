from typing import List

import torch
from torch.utils.data import Dataset

from alphazero.optimization_args import ModelingArgs
from alphazero.data.metadata import SelfPlayMetadata
from util.torch_util import Shape


class GamesDataset(Dataset):
    def __init__(self, self_play_data_dir: str, first_gen=0):
        self.self_play_metadata = SelfPlayMetadata(self_play_data_dir, first_gen=first_gen)
        self.n_total_games = self.self_play_metadata.n_total_games
        self.n_total_positions = self.self_play_metadata.n_total_positions
        self.n_window = compute_n_window(self.n_total_positions)
        self.window = self.self_play_metadata.get_window(self.n_window)
        self.key_order: List[str] = []

    def get_input_shape(self) -> Shape:
        for position_metadata in self.window:
            game_metadata = position_metadata.game_metadata
            data = torch.jit.load(game_metadata.filename).state_dict()
            return data['input'].shape[1:]
        raise Exception('Could not determine input shape!')

    def get_target_names(self) -> List[str]:
        for position_metadata in self.window:
            game_metadata = position_metadata.game_metadata
            data = torch.jit.load(game_metadata.filename).state_dict()
            names = list(data.keys())
            return [n for n in names if n != 'input']
        raise Exception('Could not extract target names!')

    def set_key_order(self, target_names: List[str]):
        self.key_order = ['input'] + target_names

    def __len__(self):
        return len(self.window)

    def __getitem__(self, idx):
        position_metadata = self.window[idx]
        game_metadata = position_metadata.game_metadata
        # COMMENT: How is position index selected?
        p = position_metadata.position_index
        try:
            data = torch.jit.load(game_metadata.filename).state_dict()
        except:
            raise Exception('Could not load data from file: {}'.format(game_metadata.filename))
        return [data[key][p] for key in self.key_order]


def compute_n_window(n_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    if ModelingArgs.fixed_window_n:
        return ModelingArgs.fixed_window_n

    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))
