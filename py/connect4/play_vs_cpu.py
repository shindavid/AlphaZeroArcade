#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

from game import Color, Game, NUM_COLUMNS, PRETTY_COLORS
from neural_net import Net, HistoryBuffer, NetWrapper, GameState

sys.path.append(os.path.join(sys.path[0], '..'))
from mcts import MCTSParams, MCTS


class Args:
    model_file: str = 'c4_model.pt'
    debug_file: str = None
    verbose: bool = False
    neural_network_only: bool = False
    num_mcts_iters: int = 100
    softmax_temperature: float = 1.0

    @staticmethod
    def load(args):
        assert args.softmax_temperature > 0
        Args.model_file = args.model_file
        Args.debug_file = args.debug_file
        Args.verbose = args.verbose
        Args.neural_network_only = args.neural_network_only
        Args.num_mcts_iters = args.num_mcts_iters
        Args.softmax_temperature = args.softmax_temperature


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", default=Args.model_file,
                        help='model output location (default: %(default)s)')
    parser.add_argument("-d", "--debug-file", help='debug output file')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose mode')
    parser.add_argument("-o", "--neural-network-only", action='store_true', help='neural network only')
    parser.add_argument("-n", "--num-mcts-iters", default=Args.num_mcts_iters, type=int,
                        help='num mcts iterations to do per move (default: %(default)s)')
    parser.add_argument("-t", "--softmax-temperature", default=Args.softmax_temperature, type=float,
                        help='softmax temperature. Must be positive. Higher=more random play (default: %(default)2f)')

    args = parser.parse_args()
    Args.load(args)


def main():
    load_args()

    print(f'Loading model from {Args.model_file}')
    model_data = torch.load(Args.model_file)
    net = Net(*model_data['model.constructor_args'])
    net.load_state_dict(model_data['model.state_dict'])
    torch.set_grad_enabled(False)
    net.eval()
    print('Model successfully loaded!')

    runner = GameRunner(net)
    runner.run()


class GameRunner:
    def __init__(self, net: Net):
        self.net = NetWrapper(net)
        self.softmax_temperature = Args.softmax_temperature
        self.neural_network_only = Args.neural_network_only
        self.verbose = Args.verbose
        self.num_previous_states = (net.input_shape[0] - 3) // 2
        self.tensor_shape = tuple([1] + list(net.input_shape))

        self.mcts = None
        self.mcts_params = None
        if not Args.neural_network_only:
            debug_file = open(Args.debug_file, 'w') if Args.debug_file else None
            self.mcts = MCTS(self.net, debug_file=debug_file)
            self.mcts_params = MCTSParams(treeSizeLimit=Args.num_mcts_iters)

        self.player_names = ['???', '???']
        self.last_move = None
        self.history_buffer = None
        self.g = Game()
        self.game_state = None

    def run(self, my_color: Optional[Color] = None):
        if my_color is None:
            my_color = np.random.randint(0, 1)
        cpu_color = 1 - my_color
        self.player_names = ['CPU', 'CPU']
        self.player_names[my_color] = 'Human'
        self.last_move = None
        self.history_buffer = HistoryBuffer(self.num_previous_states)
        self.g = Game()
        self.game_state = GameState(self.g, self.history_buffer)

        while True:
            cur_player = self.g.get_current_player()
            valid_moves = self.g.get_valid_moves()
            assert valid_moves
            if cur_player == my_color:
                result = self.handle_my_move(valid_moves)
            elif self.neural_network_only:
                result = self.handle_cpu_move_net_only(valid_moves)
            else:
                result = self.handle_cpu_move_mcts(valid_moves)

            if result is None:
                continue

            if result[my_color] == 1:
                print('Congratulations! You win!')
            elif result[cpu_color] == 1:
                print('Sorry! You lose!')
            else:
                print('The game has ended in a draw!')
            break

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
        self.game_state.applyMove(my_move-1)
        os.system('clear')
        print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        return self.game_state.getGameResult()

    def handle_cpu_move_mcts(self, valid_moves):
        results = self.mcts.sim(self.game_state, self.mcts_params)
        mcts_counts = results.counts
        heated_counts = mcts_counts.pow(1.0 / self.softmax_temperature)
        mcts_policy = heated_counts / sum(heated_counts)

        policy_prior = results.policy_prior
        value_prior = results.value_prior

        mcts_value = results.win_rates
        return self.handle_cpu_move_helper(valid_moves, policy_prior, value_prior,
                                           mcts_counts, mcts_policy, mcts_value)

    def handle_cpu_move_net_only(self, valid_moves):
        history_buffer = self.history_buffer
        tensor_shape = self.tensor_shape

        input_matrix = history_buffer.get_input()
        in_tensor = torch.reshape(torch.from_numpy(input_matrix), tensor_shape).float()

        pol_tensor, val_tensor = self.net.evaluate(in_tensor)
        pol_arr = pol_tensor.numpy()
        heated_arr = pol_arr / self.softmax_temperature
        move_probs = np.exp(heated_arr) / sum(np.exp(heated_arr))

        mask = np.zeros_like(pol_arr)
        mask[np.array(valid_moves, dtype=int) - 1] = 1
        move_probs *= mask
        move_probs /= sum(move_probs)
        value = val_tensor.softmax(dim=0)
        return self.handle_cpu_move_helper(valid_moves, move_probs, value)

    def handle_cpu_move_helper(self, valid_moves, net_policy, net_value,
                               mcts_counts=None, mcts_policy=None, mcts_value=None):
        g = self.g
        player_names = self.player_names

        policy = net_policy if mcts_policy is None else mcts_policy
        value = net_value if mcts_value is None else mcts_value

        # https://stackoverflow.com/a/65384032/543913
        policy = np.asarray(policy).astype('float64')
        policy /= np.sum(policy)

        cpu_move = np.random.choice(NUM_COLUMNS, p=policy) + 1
        assert cpu_move in valid_moves

        self.last_move = cpu_move
        self.game_state.applyMove(cpu_move-1)
        os.system('clear')
        print(g.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        if self.verbose:
            win_probs = value
            print('CPU pos eval:')
            if mcts_value is None:
                for c in (Game.RED, Game.YELLOW):
                    print('%s: %6.3f%%' % (PRETTY_COLORS[c], 100.0 * float(win_probs[c])))
                print('')
                print('%3s %8s' % ('Col', 'Net'))
                for i, x in enumerate(policy):
                    print(f'{i+1:3d} {x:8.3f}')
            else:
                net_win_probs = net_value
                for c in (Game.RED, Game.YELLOW):
                    print('%s: %6.3f%% -> %6.3f%%' % (
                        PRETTY_COLORS[c], 100.0 * float(net_win_probs[c]), 100.0 * float(win_probs[c])
                    ))
                print('')
                print('%3s %8s %8s %8s' % ('Col', 'Net', 'Count', 'MCTS'))
                for i, x in enumerate(net_policy):
                    print(f'{i+1:3d} {x:8.3f} {mcts_counts[i]:8d} {mcts_policy[i]:8.3f}')

            print('')

        return self.game_state.getGameResult()


if __name__ == '__main__':
    main()
