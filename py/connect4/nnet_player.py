import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from connect4.game_logic import C4GameState, NUM_COLUMNS, PRETTY_COLORS
from connect4.neural_net import Net, C4Tensorizor

from interface import AbstractPlayer, PlayerIndex, ActionIndex, GameResult, ActionMask
from mcts import MCTS, MCTSParams


@dataclass
class NNetPlayerParams:
    model_file: str = 'c4_model.pt'
    debug_filename: str = None
    verbose: bool = False
    neural_network_only: bool = False
    num_mcts_iters: int = 100  # set to 0 for no mcts
    temperature: float = 0.0


class NNetPlayer(AbstractPlayer):
    def __init__(self, params: NNetPlayerParams):
        self.my_index: PlayerIndex = -1
        self.params = params
        self.last_action: Optional[ActionIndex] = None
        self.net = self.load_net()
        self.num_previous_states = C4Tensorizor.get_num_previous_states(self.net.input_shape)
        self.tensor_shape = tuple([1] + list(self.net.input_shape))
        self.tensorizor = C4Tensorizor(self.num_previous_states)

        self.mcts = None
        self.mcts_params = None
        if not self.params.neural_network_only:
            self.mcts = MCTS(self.net, debug_filename=params.debug_filename)
            self.mcts_params = MCTSParams(treeSizeLimit=params.num_mcts_iters, dirichlet_mult=0)

        self.verbose_info = {}

    def load_net(self):
        model_file = self.params.model_file
        print(f'Loading model from {model_file}')
        model_data = torch.load(model_file)
        net = Net(*model_data['model.constructor_args'])
        net.load_state_dict(model_data['model.state_dict'])
        torch.set_grad_enabled(False)
        net.eval()
        print('Model successfully loaded!')
        return net

    def get_name(self) -> str:
        return 'CPU'

    def start_game(self, players: List[AbstractPlayer], seat_assignment: PlayerIndex):
        self.my_index = seat_assignment

    def receive_state_change(self, p: PlayerIndex, state: C4GameState,
                             action_index: ActionIndex, result: GameResult):
        self.tensorizor.receive_state_change(state, action_index)
        if self.mcts:
            self.mcts.receive_state_change(p, state, action_index, result)
        self.last_action = action_index

        if self.my_index == p and self.params.verbose:
            self.verbose_dump()

    def verbose_dump(self):
        if not self.verbose_info:
            return

        value = self.verbose_info['value']
        mcts_value = self.verbose_info['mcts_value']
        net_value = self.verbose_info['net_value']
        policy = self.verbose_info['policy']
        mcts_policy = self.verbose_info['mcts_policy']
        mcts_counts = self.verbose_info['mcts_counts']
        net_policy = self.verbose_info['net_policy']

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

    def get_net_only_action(self, state: C4GameState, valid_actions: ActionMask) -> ActionIndex:
        input_matrix = self.tensorizor.vectorize(state)
        in_tensor = torch.reshape(input_matrix, self.tensor_shape).float()
        transform = random.choice(self.tensorizor.get_symmetries(state))
        in_tensor = transform.transform_input(in_tensor)

        logit_policy, logit_value = [t.flatten() for t in self.net(in_tensor)]
        logit_policy = transform.transform_policy(logit_policy)

        if self.params.temperature:
            heated_policy = logit_policy / self.params.temperature
            policy = heated_policy.softmax(dim=0)
        else:
            policy = (logit_policy == logit_policy.max()).float()

        mask = torch.zeros_like(policy)
        mask[valid_actions] = 1
        policy *= mask
        policy /= sum(policy)
        assert not torch.any(policy.isnan()), (logit_policy, policy)
        value = logit_value.softmax(dim=0)
        return self.get_action_helper(policy, value)

    def get_mcts_action(self, state: C4GameState) -> ActionIndex:
        results = self.mcts.sim(self.tensorizor, state, self.mcts_params)
        mcts_counts = results.counts
        if self.params.temperature:
            heated_counts = mcts_counts.pow(1.0 / self.params.temperature)
            mcts_policy = heated_counts / sum(heated_counts)
        else:
            mcts_policy = (mcts_counts == mcts_counts.max()).float()
            mcts_policy /= mcts_policy.sum()

        policy_prior = results.policy_prior
        value_prior = results.value_prior

        mcts_value = results.win_rates
        return self.get_action_helper(policy_prior, value_prior, mcts_counts, mcts_policy, mcts_value)

    def get_action(self, state: C4GameState, valid_actions: ActionMask) -> ActionIndex:
        if self.params.neural_network_only:
            return self.get_net_only_action(state, valid_actions)
        else:
            return self.get_mcts_action(state)

    def get_action_helper(self, net_policy, net_value,
                          mcts_counts=None, mcts_policy=None, mcts_value=None):
        policy = net_policy if mcts_policy is None else mcts_policy
        value = net_value if mcts_value is None else mcts_value

        # https://stackoverflow.com/a/65384032/543913
        policy = np.asarray(policy).astype('float64')
        policy /= np.sum(policy)

        if self.params.verbose:
            self.verbose_info = {
                'value': value,
                'mcts_value': mcts_value,
                'net_value': net_value,
                'policy': policy,
                'mcts_policy': mcts_policy,
                'mcts_counts': mcts_counts,
                'net_policy': net_policy,
            }

        return np.random.choice(NUM_COLUMNS, p=policy)
