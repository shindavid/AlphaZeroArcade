#!/usr/bin/env python3
import argparse
from typing import Tuple

import h5py
from mpi4py import MPI
import numpy as np
import os
import random
import subprocess
from tqdm import tqdm

import game_logic
from connect4.game_logic import C4GameState
from connect4.neural_net import C4Tensorizor
from game_logic import NUM_ROWS, NUM_COLUMNS
from neural_net import HistoryBuffer

RANK = MPI.COMM_WORLD.Get_rank()
SIZE = MPI.COMM_WORLD.Get_size()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--c4-solver-dir', required=True,
                        help="base directory containing c4solver bin and 7x6.book")
    parser.add_argument("-n", "--num-training-games", type=int, help="number of training games")
    parser.add_argument("-g", "--games-dir", default="c4_games")
    parser.add_argument("-s", "--num-previous-states", type=int, default=0,
                        help='how many previous board states to use')

    return parser.parse_args()


def main():
    args = get_args()

    if not RANK:
        if os.path.exists(args.games_dir):
            os.system(f'rm -rf {args.games_dir}')
        os.makedirs(args.games_dir, exist_ok=True)

    MPI.COMM_WORLD.barrier()
    launch(args)


Shape = Tuple[int, ...]


def chunkify(shape: Shape, size=1024) -> Shape:
    values = list(shape)
    values[0] = min(values[0], size)
    return tuple(values)


def launch(args):
    verbose = not RANK
    num_previous_states = args.num_previous_states
    c4_solver_dir = os.path.expanduser(args.c4_solver_dir)
    c4_solver_bin = os.path.join(c4_solver_dir, 'c4solver')
    c4_solver_book = os.path.join(c4_solver_dir, '7x6.book')
    assert os.path.isdir(c4_solver_dir)
    assert os.path.isfile(c4_solver_bin)
    assert os.path.isfile(c4_solver_book)
    c4_cmd = f"{c4_solver_bin} -b {c4_solver_book} -a"
    proc = subprocess.Popen(c4_cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, encoding='utf-8')

    output_filename = os.path.join(args.games_dir, f'{RANK}.h5')
    n = args.num_training_games
    start = RANK * n // SIZE
    end = (RANK+1) * n // SIZE
    num_my_games = end - start
    max_rows = num_my_games * NUM_ROWS * NUM_COLUMNS

    NUM_PLAYERS = 2

    with h5py.File(output_filename, 'w') as output_file:
        input_shape = tuple([max_rows] + list(C4Tensorizor.get_input_shape(num_previous_states)))
        value_output_shape = (max_rows, NUM_PLAYERS)
        policy_output_shape = (max_rows, NUM_COLUMNS)

        write_index = 0
        input_dataset = output_file.create_dataset('input', shape=input_shape, dtype=np.float32,
                                                   chunks=chunkify(input_shape))
        value_dataset = output_file.create_dataset('value', shape=value_output_shape, dtype=np.float32,
                                                   chunks=chunkify(value_output_shape))
        strong_policy_dataset = output_file.create_dataset('strong_policy', shape=policy_output_shape,
                                                           dtype=np.float32, chunks=chunkify(policy_output_shape))
        weak_policy_dataset = output_file.create_dataset('weak_policy', shape=policy_output_shape,
                                                         dtype=np.float32, chunks=chunkify(policy_output_shape))

        tqdm_range = tqdm(range(num_my_games), desc="Writing training games") if verbose else range(num_my_games)
        for _ in tqdm_range:
            state = game_logic.C4GameState()
            tensorizor = C4Tensorizor(num_previous_states)
            move_history = ''
            while True:
                proc.stdin.write(move_history + '\n')
                proc.stdin.flush()
                stdout = proc.stdout.readline()

                move_scores = list(map(int, stdout.split()[-NUM_COLUMNS:]))

                cp = state.get_current_player()
                best_score = max(move_scores)
                best_move_arr = (np.array(move_scores) == best_score)
                winning_move_arr = np.array(move_scores) > 0
                cur_player_value = 0.5 * (np.sign(best_score) + 1)

                value_arr = np.zeros(2)
                value_arr[cp] = cur_player_value
                value_arr[1-cp] = 1 - cur_player_value

                input_matrix = tensorizor.vectorize()

                input_dataset[write_index] = input_matrix
                value_dataset[write_index] = value_arr
                strong_policy_dataset[write_index] = best_move_arr
                weak_policy_dataset[write_index] = winning_move_arr
                write_index += 1

                moves = state.get_valid_moves()
                assert moves
                move = random.choice(moves)
                results = state.apply_move(move-1)
                tensorizor.receive_state_change(state)
                if results is not None:
                    break

                move_history += str(move)

        input_dataset.resize(write_index, axis=0)
        value_dataset.resize(write_index, axis=0)
        strong_policy_dataset.resize(write_index, axis=0)
        weak_policy_dataset.resize(write_index, axis=0)

    if verbose:
        generic_output_filename = os.path.join(args.games_dir, '*.h5')
        print('')
        print(f'Wrote to: {generic_output_filename}')


if __name__ == '__main__':
    main()