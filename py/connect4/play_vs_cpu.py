#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from game_runner import GameRunner
from connect4.perfect_player import PerfectPlayerParams, PerfectPlayer
from connect4.game_logic import Color, C4GameState
from connect4.human_tui_player import C4HumanTuiPlayer
from connect4.nnet_player import NNetPlayer, NNetPlayerParams


class Args:
    model_file: str = 'c4_model.pt'
    debug_filename: str = None
    verbose: bool = False
    perfect: bool = False
    my_starting_color: Optional[Color] = None
    neural_network_only: bool = False
    num_mcts_iters: int = 100
    temperature: float = 0.0

    @staticmethod
    def str_to_color(s: Optional[str]) -> Optional[Color]:
        if s is None:
            return None
        if s == 'R':
            return C4GameState.RED
        if s == 'Y':
            return C4GameState.YELLOW
        raise Exception(f'Invalid color -c {s}')

    @staticmethod
    def load(args):
        Args.model_file = args.model_file
        Args.debug_filename = args.debug_filename
        Args.verbose = args.verbose
        Args.perfect = args.perfect
        Args.my_starting_color = Args.str_to_color(args.my_starting_color)
        Args.neural_network_only = args.neural_network_only
        Args.num_mcts_iters = args.num_mcts_iters
        Args.temperature = args.temperature


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", default=Args.model_file,
                        help='model output location (default: %(default)s)')
    parser.add_argument("-d", "--debug-filename", help='debug output file')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose mode')
    parser.add_argument("-p", "--perfect", action='store_true', help='play against perfect player')
    parser.add_argument("-c", "--my-starting-color", help='my starting color (R or Y). Default: random')
    parser.add_argument("-o", "--neural-network-only", action='store_true', help='neural network only')
    parser.add_argument("-n", "--num-mcts-iters", default=Args.num_mcts_iters, type=int,
                        help='num mcts iterations to do per move (default: %(default)s)')
    parser.add_argument("-t", "--temperature", default=Args.temperature, type=float,
                        help='temperature. Must be >=0. Higher=more random play (default: %(default)2f)')

    args = parser.parse_args()
    Args.load(args)


def main():
    load_args()

    human = C4HumanTuiPlayer()

    if Args.perfect:
        cpu = PerfectPlayer(PerfectPlayerParams())
    else:
        params = NNetPlayerParams(
            model_file=Args.model_file,
            debug_filename=Args.debug_filename,
            verbose=Args.verbose,
            neural_network_only=Args.neural_network_only,
            num_mcts_iters=Args.num_mcts_iters,
            temperature=Args.temperature)
        cpu = NNetPlayer(params)

    my_color = np.random.randint(0, 2) if Args.my_starting_color is None else Args.my_starting_color
    cpu_color = 1 - my_color
    players = [None, None]
    players[my_color] = human
    players[cpu_color] = cpu

    runner = GameRunner(C4GameState, players)
    result = runner.run()

    if result[my_color] == 1:
        print('Congratulations! You win!')
    elif result[cpu_color] == 1:
        print('Sorry! You lose!')
    else:
        print('The game has ended in a draw!')


if __name__ == '__main__':
    main()
