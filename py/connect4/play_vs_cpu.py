#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

from game_logic import Color, C4GameState, NUM_COLUMNS, PRETTY_COLORS
from neural_net import Net, NetWrapper, C4Tensorizor

sys.path.append(os.path.join(sys.path[0], '..'))
from mcts import MCTSParams, MCTS


class Args:
    model_file: str = 'c4_model.pt'
    debug_filename: str = None
    verbose: bool = False
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

    print(f'Loading model from {Args.model_file}')
    model_data = torch.load(Args.model_file)
    net = Net(*model_data['model.constructor_args'])
    net.load_state_dict(model_data['model.state_dict'])
    torch.set_grad_enabled(False)
    net.eval()
    print('Model successfully loaded!')

    runner = GameRunner(net)
    runner.run(Args.my_starting_color)


#class GamePlayer:
#    def play_move(self, ):

class GameRunner:
    def __init__(self, net: Net):
        self.net = NetWrapper(net)
        self.temperature = Args.temperature
        self.neural_network_only = Args.neural_network_only
        self.verbose = Args.verbose
        self.num_previous_states = C4Tensorizor.get_num_previous_states(net.input_shape)
        self.tensor_shape = tuple([1] + list(net.input_shape))

        self.player_names = ['???', '???']
        self.last_move = None
        self.history_buffer = None
        self.state = C4GameState()
        self.tensorizor = C4Tensorizor(self.num_previous_states)

        self.mcts = None
        self.mcts_params = None
        if not Args.neural_network_only:
            self.mcts = MCTS(self.net, debug_filename=Args.debug_filename)
            self.mcts_params = MCTSParams(treeSizeLimit=Args.num_mcts_iters, dirichlet_mult=0)

    def run(self, my_color: Optional[Color] = None):
        if my_color is None:
            my_color = np.random.randint(0, 2)
        cpu_color = 1 - my_color
        self.player_names = ['CPU', 'CPU']
        self.player_names[my_color] = 'Human'
        self.last_move = None
        self.state = C4GameState()
        self.tensorizor = C4Tensorizor(self.num_previous_states)

        while True:
            cur_player = self.state.get_current_player()
            valid_moves = self.state.get_valid_moves()
            assert valid_moves
            if cur_player == my_color:
                result = self.handle_my_move(valid_moves)
            elif self.neural_network_only:
                result = self.handle_cpu_move_net_only(valid_moves)
            else:
                result = self.handle_cpu_move_mcts(valid_moves)

            if result is None:
                continue

            if self.mcts is not None:
                self.mcts.record_final_position(self.state)
            if result[my_color] == 1:
                print('Congratulations! You win!')
            elif result[cpu_color] == 1:
                print('Sorry! You lose!')
            else:
                print('The game has ended in a draw!')
            break

        self.mcts.close_debug_file()
        continue_decision = input('Play again? [Y/n]: ')
        if continue_decision in ('', 'y', 'Y'):
            self.run(cpu_color)
        else:
            print('Thank you for playing! Good-bye!')

    def handle_my_move(self, valid_moves):
        game = self.state
        player_names = self.player_names

        my_move = None
        while True:
            if my_move is not None:
                os.system('clear')
                print(game.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
                print(f'Invalid input!')
            my_move = input('Enter move [1-7]: ')
            try:
                my_move = int(my_move)
                assert my_move in valid_moves
                break
            except:
                continue

        self.last_move = my_move
        action_index = my_move - 1
        result = self.state.apply_move(action_index)
        self.tensorizor.receive_state_change(self.state, action_index)
        os.system('clear')
        print(game.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        return result

    def handle_cpu_move_mcts(self, valid_moves):
        results = self.mcts.sim(self.tensorizor, self.state, self.mcts_params)
        mcts_counts = results.counts
        if self.temperature:
            heated_counts = mcts_counts.pow(1.0 / self.temperature)
            mcts_policy = heated_counts / sum(heated_counts)
        else:
            mcts_policy = (mcts_counts == mcts_counts.max()).float()
            mcts_policy /= mcts_policy.sum()

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

        logit_policy, logit_value = self.net.evaluate(in_tensor)

        if self.temperature:
            heated_policy = logit_policy / self.temperature
            policy = heated_policy.softmax(dim=0)
        else:
            policy = (logit_policy == logit_policy.max()).float()

        mask = torch.zeros_like(policy)
        mask[np.array(valid_moves, dtype=int) - 1] = 1
        policy *= mask
        policy /= sum(policy)
        assert not torch.any(policy.isnan()), (logit_policy, policy)
        value = logit_value.softmax(dim=0)
        return self.handle_cpu_move_helper(valid_moves, policy, value)

    def handle_cpu_move_helper(self, valid_moves, net_policy, net_value,
                               mcts_counts=None, mcts_policy=None, mcts_value=None):
        game = self.state
        player_names = self.player_names

        policy = net_policy if mcts_policy is None else mcts_policy
        value = net_value if mcts_value is None else mcts_value

        # https://stackoverflow.com/a/65384032/543913
        policy = np.asarray(policy).astype('float64')
        policy /= np.sum(policy)

        cpu_move = np.random.choice(NUM_COLUMNS, p=policy) + 1
        assert cpu_move in valid_moves

        self.last_move = cpu_move
        action_index = cpu_move - 1
        result = self.state.apply_move(action_index)
        self.tensorizor.receive_state_change(self.state, action_index)
        os.system('clear')
        print(game.to_ascii_drawing(add_legend=True, player_names=player_names, highlight_column=self.last_move))
        if self.verbose:
            win_probs = value
            print('CPU pos eval:')
            if mcts_value is None:
                for c in (C4GameState.RED, C4GameState.YELLOW):
                    print('%s: %6.3f%%' % (PRETTY_COLORS[c], 100.0 * float(win_probs[c])))
                print('')
                print('%3s %8s' % ('Col', 'Net'))
                for i, x in enumerate(policy):
                    print(f'{i+1:3d} {x:8.3f}')
            else:
                net_win_probs = net_value
                for c in (C4GameState.RED, C4GameState.YELLOW):
                    print('%s: %6.3f%% -> %6.3f%%' % (
                        PRETTY_COLORS[c], 100.0 * float(net_win_probs[c]), 100.0 * float(win_probs[c])
                    ))
                print('')
                print('%3s %8s %8s %8s' % ('Col', 'Net', 'Count', 'MCTS'))
                for i, x in enumerate(net_policy):
                    print(f'{i+1:3d} {x:8.3f} {mcts_counts[i]:8d} {mcts_policy[i]:8.3f}')

            print('')

        return result


if __name__ == '__main__':
    main()
