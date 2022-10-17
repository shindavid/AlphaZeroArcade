"""
Action items:

1. MCGS instead of MCTS (Tree -> Graph)? This makes sense because different ordering of moves can arrive at the same
   game state. But, the neural net takes recent history as input, which means the states aren't identical from the
   net's point of view. How do we deal with that? How do we deal with graph cycles?

2. Neural net cache.

3. Subtree reuse (how does this interact with Dirichlet noise?).

4. Root parallelism. In turn demands multiprocess setup, virtual loss, etc.
"""
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn as nn
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet

from interface import AbstractGameState, ActionIndex, ActionMask, PlayerIndex, \
    LocalPolicyLogitDistr, LocalPolicyProbDistr, ValueProbDistr, GlobalPolicyProbDistr, AbstractGameTensorizor, \
    GameResult

GlobalPolicyCountDistr = Tensor


@dataclass
class MCTSParams:
    treeSizeLimit: int
    root_softmax_temperature: float = 1.03
    c_PUCT: float = 1.1
    dirichlet_mult: float = 0.25
    dirichlet_alpha: float = 0.03


class MCTSNodeStats:
    def __init__(self):
        self.count = 0
        self.value_sum = 0
        self.policy_prior = 0.0


@dataclass
class MCTSResults:
    counts: GlobalPolicyCountDistr
    win_rates: ValueProbDistr
    policy_prior: GlobalPolicyProbDistr
    value_prior: ValueProbDistr


@dataclass
class StateEvaluation:
    def __init__(self, net: nn.Module, tensorizor: AbstractGameTensorizor, state: AbstractGameState,
                 result: GameResult):
        self.current_player: PlayerIndex = state.get_current_player()
        self.game_result = result

        self.valid_action_mask: Optional[ActionMask] = None
        self.local_policy_logit_distr: Optional[LocalPolicyLogitDistr] = None
        self.value_prob_distr: Optional[ValueProbDistr] = None

        if self.game_result is not None:
            # game is over, don't bother computing other fields
            return

        tensor_input = tensorizor.vectorize(state)
        transform = random.choice(tensorizor.get_symmetries(state))
        tensor_input = transform.transform_input(tensor_input)
        policy_output, value_output = [t.flatten() for t in net(tensor_input)]
        policy_output = transform.transform_policy(policy_output)

        self.valid_action_mask = state.get_valid_actions()
        self.local_policy_logit_distr = policy_output[torch.where(self.valid_action_mask)[0]]
        self.value_prob_distr = value_output.softmax(dim=0)


class Tree:
    def __init__(self, n_players: int, action_index: Optional[ActionIndex] = None, parent: Optional['Tree'] = None):
        self.evaluation: Optional[StateEvaluation] = None
        self.valid_action_indices: Optional[List[ActionIndex]] = None
        self.children: Optional[List[Tree]] = None
        self.action_index: Optional[ActionIndex] = action_index
        self.parent = parent
        self.count = 0
        self.value_sum = torch.zeros(n_players)
        self.value_avg = torch.zeros(n_players)
        self.value_prior: Optional[ValueProbDistr] = None
        self.policy_prior: Optional[LocalPolicyProbDistr] = None

    @property
    def n_players(self) -> int:
        return self.value_sum.shape[0]

    def is_root(self) -> bool:
        return self.parent is None

    def win_rates(self) -> ValueProbDistr:
        return self.value_sum / (self.count if self.count else 1.0)

    def has_children(self) -> bool:
        return bool(self.children)

    def expand_children(self, evaluation: StateEvaluation):
        if self.children is not None:
            return

        valid_action_mask = evaluation.valid_action_mask
        self.valid_action_indices = np.where(valid_action_mask)[0]
        self.children = [Tree(self.n_players, action_index, self) for action_index in self.valid_action_indices]

    def backprop(self, result: ValueProbDistr):
        self.count += 1
        self.value_sum += result
        self.value_avg = self.value_sum / self.count
        if self.parent:
            self.parent.backprop(result)

    def compute_policy_prior(self, evaluation: StateEvaluation, params: MCTSParams) -> LocalPolicyProbDistr:
        if self.policy_prior is not None:
            return self.policy_prior

        policy_output = evaluation.local_policy_logit_distr

        is_root = self.is_root()
        inv_temp = (1.0 / params.root_softmax_temperature) if is_root else 1.0

        P = torch.softmax(policy_output * inv_temp, dim=0)
        self.value_prior = evaluation.value_prob_distr
        self.policy_prior = P
        return P


class MCTS:
    """
    Constructed once per game.

    Some notes from Katago paper...

    At node n, choose child c that maximizes:

    PUCT(c) = V(c) + c_{PUCT}*P(c)*sqrt(sum_{c'} N(c')) / (1 + N(c))

    where:
    - V(c): avg predicted utility of all nodes in c's subtree
    - P(c): policy prior of c from net
    - N(c): # playouts previously sent thru c
    - c_{PUCT}: 1.1

    KataGo adds noise to policy prior P(c) at the root:

    P(c) = 0.75*P_{raw}(c) + 0.25*nu

    where nu is a draw from a Dirichlet distribution on legal moves with param alpha = 0.03*19^2 / N(c),
    where N is the total number of legal moves. KataGo also applies a softmax temp at the root of 1.03

    ** Playout Cap Randomization **

    - with probability p ~= 0.25, we do a FULL SEARCH:
        - tree grows to size N
        - enable Dirichlet noise + explorative settings
        - export for nnet training
    - else, we do FAST SEARCH:
    - tree grows to size n < N
    - disable Dirichlet noise + explorative settings (maximizing strength)
    - do NOT export for nnet training
    """
    def __init__(self, net: nn.Module, n_players: int = 2, debug_filename: Optional[str] = None):
        self.net = net
        self.debug_filename = debug_filename
        self.n_players = n_players
        self.root: Optional[Tree] = None
        self.cache: Dict[Any, StateEvaluation] = {}
        self.debug_tree = None if debug_filename is None else ET.ElementTree(ET.Element('Game'))
        self.player_index: Optional[PlayerIndex] = None

    def _terminal_visit_debug(
            self, player: PlayerIndex, depth: int, state: AbstractGameState, game_result: GameResult, tree: Tree,
            debug_subtree: Optional[ET.Element]):
        if not self.debug_filename:
            return
        tree_dict = {
            'player': player,
            'depth': depth,
            'board': state.compact_repr(),
            'terminal': True,
            'eval': game_result,
            'value_sum': tree.value_sum,
        }
        if tree.action_index is not None:
            tree_dict['action'] = tree.action_index
        tree_dict = {k: stringify(v) for k, v in tree_dict.items()}
        elem = debug_subtree.makeelement('Visit', tree_dict)
        debug_subtree.insert(0, elem)

    def _internal_visit_debug(
            self, player: PlayerIndex, depth: int, board: str, evaluation: StateEvaluation, leaf: bool,
            value_sum: Tensor, tree: Tree, debug_subtree: Optional[ET.Element], P: Tensor, noise: Tensor, V: Tensor,
            N: Tensor, PUCT: Tensor):
        if not self.debug_filename:
            return

        tree_dict = {
            'player': player,
            'depth': depth,
            'board': board,
            'eval': evaluation.value_prob_distr,
            'leaf': leaf,
            'value_sum': value_sum,
        }
        if tree.action_index is not None:
            tree_dict['action'] = tree.action_index
        tree_dict = {k: stringify(v) for k, v in tree_dict.items()}
        elem = debug_subtree.makeelement('Visit', tree_dict)
        debug_subtree.insert(0, elem)
        for ac, rp, p, no, v, n, puct in zip(tree.valid_action_indices, tree.policy_prior, P, noise, V, N, PUCT):
            attribs = {
                'action': ac,
                'rP': rp,
                'P': p,
                'dir': no,
                'V': v,
                'N': n,
                'PUCT': puct,
            }
            attribs = {k: stringify(v) for k, v in attribs.items()}
            ET.SubElement(elem, 'Child', attribs)

    def close_debug_file(self):
        if self.debug_filename is None:
            return
        ET.indent(self.debug_tree)
        self.debug_tree.write(self.debug_filename, encoding='utf-8', xml_declaration=True)
        self.debug_filename = None
        self.debug_tree = None

    def record_final_position(self, state: AbstractGameState):
        if self.debug_tree:
            ET.SubElement(self.debug_tree.getroot(), 'Move', board=state.compact_repr())

    def sim(self, tensorizor: AbstractGameTensorizor, state: AbstractGameState, params: MCTSParams) -> MCTSResults:
        if self.player_index is None:
            self.player_index = state.get_current_player()
            if self.debug_tree is not None:
                self.debug_tree.getroot().set('player', str(self.player_index))

        move_tree = None
        if self.debug_tree is not None:
            move_tree = ET.SubElement(self.debug_tree.getroot(), 'Move', board=state.compact_repr())
        self.root = Tree(self.n_players)

        orig_tensorizor, orig_state = tensorizor, state
        tensorizor = orig_tensorizor.clone()
        state = orig_state.clone()

        for i in range(params.treeSizeLimit):
            iter_tree = None if move_tree is None else ET.SubElement(move_tree, 'Iter', i=str(i))
            self.visit(self.root, tensorizor, state, params, 1, None, debug_subtree=iter_tree)
            if not tensorizor.supports_undo():
                tensorizor = orig_tensorizor.clone()
                state = orig_state.clone()

        # TODO: return local distrs instead of global distrs. I'm returning global for now only for debugging.
        n = state.get_num_global_actions()
        counts = torch.zeros(n, dtype=int)
        for child in self.root.children:
            counts[child.action_index] = child.count
        win_rates = self.root.win_rates()
        policy_prior = torch.zeros(n)
        policy_prior[self.root.valid_action_indices] = self.root.policy_prior
        value_prior = self.root.value_prior
        return MCTSResults(counts, win_rates, policy_prior, value_prior)

    def evaluate(self, tree: Tree, tensorizor: AbstractGameTensorizor, state: AbstractGameState,
                 result: GameResult) -> StateEvaluation:
        if tree.evaluation is not None:
            return tree.evaluation

        signature = state.get_signature()
        evaluation = self.cache.get(signature, None)
        if evaluation is None:
            evaluation = StateEvaluation(self.net, tensorizor, state,  result)
            self.cache[signature] = evaluation

        tree.evaluation = evaluation
        return evaluation

    def visit(self, tree: Tree, tensorizor: AbstractGameTensorizor, state: AbstractGameState, params: MCTSParams,
              depth: int, result: GameResult, debug_subtree: Optional[ET.Element]):
        evaluation = self.evaluate(tree, tensorizor, state, result)
        game_result = evaluation.game_result
        current_player = evaluation.current_player

        if game_result is not None:
            tree.backprop(game_result)
            self._terminal_visit_debug(current_player, depth, state, game_result, tree, debug_subtree)
            return

        leaf = not tree.has_children()
        tree.expand_children(evaluation)

        c_PUCT = params.c_PUCT
        P = tree.compute_policy_prior(evaluation, params)
        noise = torch.zeros(len(P))
        if tree.is_root() and params.dirichlet_mult:
            noise = Dirichlet([params.dirichlet_alpha] * len(P)).sample()
            P = (1.0 - params.dirichlet_mult) * P + params.dirichlet_mult * noise

        V = torch.Tensor([c.value_avg[current_player] for c in tree.children])
        N = torch.Tensor([c.count for c in tree.children])
        eps = 1e-6  # needed when N == 0
        PUCT = V + c_PUCT * P * (np.sqrt(sum(N) + eps)) / (1 + N)
        value_sum = torch.Tensor(tree.value_sum)

        best_child: Tree = tree.children[np.argmax(PUCT)]

        board = state.compact_repr() if self.debug_filename else None

        if leaf:
            tree.backprop(evaluation.value_prob_distr)
        else:
            result = state.apply_move(best_child.action_index)
            tensorizor.receive_state_change(state, best_child.action_index)
            self.visit(best_child, tensorizor, state, params, depth + 1, result, debug_subtree)

            if tensorizor.supports_undo():
                tensorizor.undo(state)

        self._internal_visit_debug(
            player=current_player, depth=depth, board=board, evaluation=evaluation, leaf=leaf, value_sum=value_sum,
            tree=tree, debug_subtree=debug_subtree, P=P, noise=noise, V=V, N=N, PUCT=PUCT)


def stringify(value):
    if isinstance(value, torch.Tensor):
        return stringify(value.tolist())
    if isinstance(value, np.ndarray):
        assert len(value.shape) == 1, value.shape
        return stringify(value.tolist())
    if isinstance(value, (list, tuple)):
        return ','.join(map(stringify, value))
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return '%.5g' % value
    return str(value)
