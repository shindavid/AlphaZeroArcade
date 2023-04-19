import torch
from torch.utils.data import Dataset

from alphazero.optimization_args import ModelingArgs
from alphazero.data.metadata import SelfPlayMetadata
from util.torch_util import Shape


class GamesDataset(Dataset):
    def __init__(self, self_play_data_dir: str):
        self.self_play_metadata = SelfPlayMetadata(self_play_data_dir)
        self.n_total_games = self.self_play_metadata.n_total_games
        self.n_total_positions = self.self_play_metadata.n_total_positions
        self.n_window = compute_n_window(self.n_total_positions)
        self.window = self.self_play_metadata.get_window(self.n_window)

    def get_input_shape(self) -> Shape:
        for position_metadata in self.window:
            game_metadata = position_metadata.game_metadata
            data = torch.jit.load(game_metadata.filename).state_dict()
            return data['input'].shape[1:]
        raise Exception('Could not determine input shape!')

    def __len__(self):
        return len(self.window)

    def __getitem__(self, idx):
        position_metadata = self.window[idx]
        game_metadata = position_metadata.game_metadata
        # COMMENT: How is position index selected?
        p = position_metadata.position_index
        data = torch.jit.load(game_metadata.filename).state_dict()
        return data['input'][p], data['value'][p], data['policy'][p]


def compute_n_window(n_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))
