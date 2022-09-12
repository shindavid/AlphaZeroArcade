#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import numpy as np
import torch

from game import Color, Game, NUM_COLUMNS, NUM_ROWS
from neural_net import InputBuilder, Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", default="c4_model.pt",
                        help='model output location (default: %(default)s)')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose mode')
    parser.add_argument("-t", "--softmax-temperature", default=1.0, type=float,
                        help='softmax temperature. Must be positive. Higher=more random play (default: %(default)2f)')

    args = parser.parse_args()
    assert args.softmax_temperature > 0
    return args


def main():
    args = get_args()
    softmax_temperature = args.softmax_temperature
    model_file = args.model_file
    verbose = args.verbose

    print(f'Loading model from {model_file}')
    model_data = torch.load(model_file)
    net = Net(*model_data['model.constructor_args'])
    net.load_state_dict(model_data['model.state_dict'])
    torch.set_grad_enabled(False)
    net.eval()
    print('Model successfully loaded!')

    run_game(net, softmax_temperature, verbose=verbose)


def run_game(net: Net, softmax_temperature: float, my_color: Optional[Color] = None, verbose=False):
    if my_color is None:
        my_color = np.random.randint(0, 1)
    cpu_color = 1 - my_color
    player_names = ['CPU', 'CPU']
    player_names[my_color] = 'Human'

    num_previous_states = (net.input_shape[0] - 3) // 2
    tensor_shape = tuple([1] + list(net.input_shape))
    builder = InputBuilder(num_previous_states)
    g = Game()

    last_move = None
    while True:
        cur_player = g.get_current_player()
        valid_moves = g.get_valid_moves()
        if not valid_moves:
            print('The game has ended in a draw!')
            break
        if cur_player == my_color:
            my_move = None
            while True:
                if my_move is not None:
                    os.system('clear')
                    print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=last_move))
                    print(f'Invalid input!')
                my_move = input('Enter move [1-7]: ')
                try:
                    my_move = int(my_move)
                    assert my_move in valid_moves
                    break
                except:
                    continue

            last_move = my_move
            winner = g.apply_move(my_move)
            os.system('clear')
            print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=last_move))
            if winner is not None:
                print('Congratulations! You win!')
                break
        else:
            input_matrix = builder.get_input(g)
            in_tensor = torch.reshape(torch.from_numpy(input_matrix), tensor_shape).float()

            pol_tensor, val_tensor = net(in_tensor)
            pol_arr = pol_tensor.numpy()[0]
            heated_arr = pol_arr / softmax_temperature
            move_probs = np.exp(heated_arr) / sum(np.exp(heated_arr))
            cpu_pos_eval = float(val_tensor)

            mask = np.zeros_like(pol_arr)
            mask[np.array(valid_moves, dtype=int) - 1] = 1
            move_probs *= mask

            move_probs /= sum(move_probs)
            cpu_move = np.random.choice(NUM_COLUMNS, p=move_probs) + 1
            assert cpu_move in valid_moves

            last_move = cpu_move
            winner = g.apply_move(cpu_move)
            os.system('clear')
            print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=last_move))
            if verbose:
                print('CPU pos eval: %.3f' % cpu_pos_eval)
                print('')

                print('%3s %8s %8s' % ('Col', 'Net', 'Prob'))
                for i, x in enumerate(pol_arr):
                    print(f'{i+1:3d} {x:+8.3f} {move_probs[i]:8.3f}')
                print('')

            if winner is not None:
                print('Sorry! You lose!')
                break

    continue_decision = input('Play again? [Y/n]: ')
    if continue_decision in ('', 'y', 'Y'):
        return run_game(net, softmax_temperature, cpu_color, verbose=verbose)
    else:
        print('Thank you for playing! Good-bye!')


if __name__ == '__main__':
    main()
