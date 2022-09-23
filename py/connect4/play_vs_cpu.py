#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

from game import Color, Game, NUM_COLUMNS, NUM_ROWS, PRETTY_COLORS
from neural_net import Net, HistoryBuffer

sys.path.append(os.path.join(sys.path[0], '..'))
import mcts


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

    runner = GameRunner(net, softmax_temperature, verbose=verbose)
    runner.run()


class GameRunner:
    def __init__(self, net: Net, softmax_temperature: float, verbose=False):
        self.net = net
        self.softmax_temperature = softmax_temperature
        self.verbose = verbose
        self.num_previous_states = (net.input_shape[0] - 3) // 2
        self.tensor_shape = tuple([1] + list(net.input_shape))

        self.player_names = ['???', '???']
        self.last_move = None
        self.history_buffer = None
        self.g = Game()

    def run(self, my_color: Optional[Color] = None):
        if my_color is None:
            my_color = np.random.randint(0, 1)
        cpu_color = 1 - my_color
        self.player_names = ['CPU', 'CPU']
        self.player_names[my_color] = 'Human'
        self.last_move = None
        self.history_buffer = HistoryBuffer(self.num_previous_states)
        self.g = Game()

        while True:
            cur_player = self.g.get_current_player()
            valid_moves = self.g.get_valid_moves()
            if not valid_moves:
                print('The game has ended in a draw!')
                break
            if cur_player == my_color:
                if self.handle_my_move(valid_moves):
                    break
            else:
                if self.handle_cpu_move(valid_moves):
                    break

            self.history_buffer.update(self.g)

        continue_decision = input('Play again? [Y/n]: ')
        if continue_decision in ('', 'y', 'Y'):
            self.run(cpu_color)
        else:
            print('Thank you for playing! Good-bye!')

    def handle_my_move(self, valid_moves):
        g = self.g
        player_names = self.player_names

        my_move = None
        while True:
            if my_move is not None:
                os.system('clear')
                print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
                print(f'Invalid input!')
            my_move = input('Enter move [1-7]: ')
            try:
                my_move = int(my_move)
                assert my_move in valid_moves
                break
            except:
                continue

        self.last_move = my_move
        winner = g.apply_move(my_move)
        os.system('clear')
        print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        if winner is not None:
            print('Congratulations! You win!')
            return True
        return False

    def handle_cpu_move(self, valid_moves):
        history_buffer = self.history_buffer
        tensor_shape = self.tensor_shape
        net = self.net
        softmax_temperature = self.softmax_temperature
        verbose = self.verbose
        g = self.g
        player_names = self.player_names

        input_matrix = history_buffer.get_input()
        in_tensor = torch.reshape(torch.from_numpy(input_matrix), tensor_shape).float()

        pol_tensor, val_tensor = net(in_tensor)
        pol_arr = pol_tensor.numpy()[0]
        heated_arr = pol_arr / softmax_temperature
        move_probs = np.exp(heated_arr) / sum(np.exp(heated_arr))

        mask = np.zeros_like(pol_arr)
        mask[np.array(valid_moves, dtype=int) - 1] = 1
        move_probs *= mask

        move_probs /= sum(move_probs)
        cpu_move = np.random.choice(NUM_COLUMNS, p=move_probs) + 1
        assert cpu_move in valid_moves

        self.last_move = cpu_move
        winner = g.apply_move(cpu_move)
        os.system('clear')
        print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        if verbose:
            win_probs = val_tensor.softmax(dim=1).flatten()
            print('CPU pos eval:')
            for c in (Game.RED, Game.YELLOW):
                print('%s: %6.3f%%' % (PRETTY_COLORS[c], 100.0 * float(win_probs[c])))
            print('')

            print('%3s %8s %8s' % ('Col', 'Net', 'Prob'))
            for i, x in enumerate(pol_arr):
                print(f'{i+1:3d} {x:+8.3f} {move_probs[i]:8.3f}')
            print('')

        if winner is not None:
            print('Sorry! You lose!')
            return True
        return False


if __name__ == '__main__':
    main()
