#!/usr/bin/env python3

import argparse
import os
import random
from typing import Dict, List

from alphazero import shared
from config import Config


class Args:
    c4_base_dir: str

    @staticmethod
    def load(args):
        Args.c4_base_dir = args.c4_base_dir
        assert Args.c4_base_dir, 'Required option: -d'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()
    cfg.add_parser_argument('c4.base_dir', parser, '-d', '--c4-base-dir', help='base-dir for game/model files')
    shared.add_optimization_args(parser)

    args = parser.parse_args()
    Args.load(args)
    shared.OptimizationArgs.load(args)


Generation = int


class SelfPlayGameMetadata:
    def __init__(self, filename: str):
        self.filename = filename
        info = os.path.split(filename)[1].split('.')[0].split('-')  # 1685860410604914-10.pt
        self.timestamp = int(info[0])
        self.num_positions = int(info[1])


class GenerationMetadata:
    def __init__(self, full_gen_dir: str):
        self.game_metadata_list = []
        for filename in os.listdir(full_gen_dir):
            if filename.startswith('.'):
                continue
            full_filename = os.path.join(full_gen_dir, filename)
            game_metadata = SelfPlayGameMetadata(full_filename)
            self.game_metadata_list.append(game_metadata)

        self.game_metadata_list.sort(key=lambda g: -g.timestamp)  # sort reverse chronological order
        self.num_positions = sum(g.num_positions for g in self.game_metadata_list)


def compute_n_window(N_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = shared.OptimizationArgs.window_c
    alpha = shared.OptimizationArgs.window_alpha
    beta = shared.OptimizationArgs.window_beta
    return min(N_total, int(c * (1 + beta * ((N_total / c) ** alpha - 1) / alpha)))


class SelfPlayMetadata:
    def __init__(self, self_play_dir: str):
        self.self_play_dir = self_play_dir
        self.metadata: Dict[Generation, GenerationMetadata] = {}
        self.n_total_positions = 0
        for gen_dir in os.listdir(self_play_dir):
            assert gen_dir.startswith('gen'), gen_dir
            generation = int(gen_dir[3:])
            full_gen_dir = os.path.join(self_play_dir, gen_dir)
            metadata = GenerationMetadata(full_gen_dir)
            self.metadata[generation] = metadata
            self.n_total_positions += metadata.num_positions

    def get_window(self, n_window: int) -> List[SelfPlayGameMetadata]:
        window = []
        cumulative_n_games = 0
        for generation in reversed(self.metadata.keys()):
            gen_metadata = self.metadata[generation]
            n = len(gen_metadata.game_metadata_list)
            i = 0
            while cumulative_n_games < n_window and i < n:
                game_metadata = gen_metadata.game_metadata_list[i]
                cumulative_n_games += game_metadata.num_positions
                i += 1
                window.append(game_metadata)
        return window


class C4DataLoader:
    def __init__(self, manager: shared.AlphaZeroManager):
        self.manager = manager
        self.self_play_metadata = SelfPlayMetadata(manager.self_play_dir)
        n_total = self.self_play_metadata.n_total_positions
        n_window = compute_n_window(n_total)
        print(f'Sampling from the {n_window} most recent positions among {n_total} total positions')
        self.window = self.self_play_metadata.get_window(n_window)
        random.shuffle(self.window)


def main():
    load_args()
    manager = shared.AlphaZeroManager(Args.c4_base_dir)
    loader = C4DataLoader(manager)




if __name__ == '__main__':
    main()
