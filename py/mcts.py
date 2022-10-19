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
from neural_net import ENABLE_CUDA
from profiling import ProfilerRegistry

GlobalPolicyCountDistr = Tensor


@dataclass
class MCTSParams:
    treeSizeLimit: int
    root_softmax_temperature: float = 1.03
    c_PUCT: float = 1.1
    dirichlet_mult: float = 0.25
    dirichlet_alpha: float = 0.03
    allow_eliminations: bool = True

    def can_reuse_subtree(self) -> bool:
        return not bool(self.dirichlet_mult)


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
        ProfilerRegistry['net.eval'].start()
        policy_output, value_output = net(tensor_input)
        ProfilerRegistry['net.eval'].stop()
        if ENABLE_CUDA:
            ProfilerRegistry['gpu.transfer2'].start()
            policy_output = policy_output.to('cpu')
            value_output = value_output.to('cpu')
            ProfilerRegistry['gpu.transfer2'].stop()

        policy_output = policy_output.flatten()
        value_output = value_output.flatten()
        policy_output = transform.transform_policy(policy_output)

        self.valid_action_mask = state.get_valid_actions()
        self.local_policy_logit_distr = policy_output[torch.where(self.valid_action_mask)[0]]
        self.value_prob_distr = value_output.softmax(dim=0)


class Tree:
    """
    Implementation notes:

    ** Terminal states **

    When we reach a terminal game state, we have certainty over the game result, but vanilla MCTS does not provide
    the mechanics to reflect this certainty. This implementation achieves the certainty by maintaining a member
    :self.V_floor:, which tracks a provable score floor for each player from the given node. This is back-propagated up
    the tree in minimax fashion. When a node has :max(self.V_floor)==1:, it is a provably won or lost position, and so
    we trim it from the tree, pretending that corresponding visits to that node never happened.
    """
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

        self.V_floor = torch.zeros(n_players)
        self.eliminated = False

    def get_effective_visit_counts(self, num_global_actions) -> GlobalPolicyCountDistr:
        current_player = self.evaluation.current_player
        counts = torch.zeros(num_global_actions, dtype=int)

        if self.eliminated:
            # identify the child(ren) with maximal V_floor, and put all counts there
            max_V_floor = max(c.V_floor[current_player] for c in self.children)
            for child in self.children:
                if child.V_floor[current_player] == max_V_floor:
                    counts[child.action_index] = 1
            return counts

        for child in self.children:
            counts[child.action_index] = child.effective_count()
        return counts

    def effective_count(self) -> int:
        return 0 if self.eliminated else self.count

    def effective_value_avg(self, p: PlayerIndex):
        return self.V_floor[p] if self.has_certain_outcome() else self.value_avg[p]

    @property
    def n_players(self) -> int:
        return self.value_sum.shape[0]

    def is_root(self) -> bool:
        return self.parent is None

    def win_rates(self) -> ValueProbDistr:
        return self.value_sum / (self.count if self.count else 1.0)

    def has_children(self) -> bool:
        return bool(self.children)

    def is_leaf(self) -> bool:
        return not self.has_children()

    def expand_children(self, evaluation: StateEvaluation):
        if self.children is not None:
            return

        valid_action_mask = evaluation.valid_action_mask
        self.valid_action_indices = np.where(valid_action_mask)[0]
        self.children = [Tree(self.n_players, action_index, self) for action_index in self.valid_action_indices]

    def has_certain_outcome(self) -> bool:
        """
        This includes certain wins/losses AND certain draws.
        """
        return float(sum(self.V_floor)) == 1

    def can_be_eliminated(self) -> bool:
        """
        We only eliminate won or lost positions, not drawn positions.

        Drawn positions are not eliminated because MCTS still needs some way to compare a provably-drawn position vs an
        uncertain position. It needs to accumulate visits to provably-drawn positions to do this.
        """
        return float(max(self.V_floor)) == 1

    def backprop(self, result: ValueProbDistr, terminal: bool = False):
        self.count += 1
        self.value_sum += result
        self.value_avg = self.value_sum / self.count
        if self.parent:
            self.parent.backprop(result)

        if terminal:
            self.terminal_backprop(result)

    def terminal_backprop(self, result: GameResult = None):
        if result is None:
            current_player = self.evaluation.current_player
            for p in range(self.n_players):
                minimax = max if p == current_player else min
                self.V_floor[p] = minimax(c.V_floor[p] for c in self.children)
        else:
            self.V_floor = result

        if self.can_be_eliminated():
            self.eliminated = True
            if self.parent:
                self.parent.terminal_backprop()

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

    def clear(self):
        self.root = None
        self.cache = {}

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
            'value_avg': tree.value_sum,
        }
        if tree.action_index is not None:
            tree_dict['action'] = tree.action_index
        tree_dict = {k: stringify(v) for k, v in tree_dict.items()}
        elem = debug_subtree.makeelement('Visit', tree_dict)
        debug_subtree.insert(0, elem)

    def _internal_visit_debug(
            self, player: PlayerIndex, depth: int, board: str, evaluation: StateEvaluation, leaf: bool,
            value_avg: Tensor, tree: Tree, debug_subtree: Optional[ET.Element], P: Tensor, noise: Tensor, V: Tensor,
            N: Tensor, PUCT: Tensor, E: Tensor):
        if not self.debug_filename:
            return

        tree_dict = {
            'player': player,
            'depth': depth,
            'board': board,
            'eval': evaluation.value_prob_distr,
            'leaf': leaf,
            'value_avg': value_avg,
        }
        if tree.action_index is not None:
            tree_dict['action'] = tree.action_index
        tree_dict = {k: stringify(v) for k, v in tree_dict.items()}
        elem = debug_subtree.makeelement('Visit', tree_dict)
        debug_subtree.insert(0, elem)
        for ac, rp, p, no, v, n, puct, e in zip(tree.valid_action_indices, tree.policy_prior, P, noise, V, N, PUCT, E):
            attribs = {
                'action': ac,
                'rP': rp,
                'P': p,
                'dir': no,
                'V': v,
                'N': n,
                'E': e,
                'PUCT': puct,
            }
            attribs = {key: stringify(value) for key, value in attribs.items()}
            ET.SubElement(elem, 'Child', attribs)

    def receive_state_change(self, p: PlayerIndex, state: AbstractGameState,
                             action_index: ActionIndex, result: GameResult):
        if self.root:
            root = self.root
            assert root.evaluation.current_player == p
            self.root = None
            if root.children:
                matching_children = [c for c in root.children if c.action_index == action_index]
                assert len(matching_children) <= 1
                if matching_children:
                    child = matching_children[0]
                    self.root = child

        if result is not None and self.debug_tree:
            ET.SubElement(self.debug_tree.getroot(), 'Move', board=state.compact_repr())
            ET.indent(self.debug_tree)
            self.debug_tree.write(self.debug_filename, encoding='utf-8', xml_declaration=True)
            self.debug_filename = None
            self.debug_tree = None

    def sim(self, tensorizor: AbstractGameTensorizor, state: AbstractGameState, params: MCTSParams) -> MCTSResults:
        assert torch.cuda.is_available()  # cuda sometimes randomly disables, don't want slow-ass runs

        if self.player_index is None:
            self.player_index = state.get_current_player()
            if self.debug_tree is not None:
                self.debug_tree.getroot().set('player', str(self.player_index))

        move_tree = None
        if self.debug_tree is not None:
            move_tree = ET.SubElement(self.debug_tree.getroot(), 'Move', board=state.compact_repr())

        if not params.can_reuse_subtree() or self.root is None:
            self.root = Tree(self.n_players)

        orig_tensorizor, orig_state = tensorizor, state
        tensorizor = orig_tensorizor.clone()
        state = orig_state.clone()

        i = 0
        while self.root.effective_count() < params.treeSizeLimit and not self.root.eliminated:
            iter_tree = None if move_tree is None else ET.SubElement(move_tree, 'Iter', i=str(i))
            i += 1
            self.visit(self.root, tensorizor, state, params, 1, None, debug_subtree=iter_tree)
            if not tensorizor.supports_undo():
                tensorizor = orig_tensorizor.clone()
                state = orig_state.clone()
            i += 1

        # TODO: return local distrs instead of global distrs. I'm returning global for now only for debugging.
        n = state.get_num_global_actions()
        counts = self.root.get_effective_visit_counts(n)
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
            tree.backprop(game_result, terminal=params.allow_eliminations)
            self._terminal_visit_debug(current_player, depth, state, game_result, tree, debug_subtree)
            return

        leaf = tree.is_leaf()
        tree.expand_children(evaluation)

        c_PUCT = params.c_PUCT
        P = tree.compute_policy_prior(evaluation, params)
        noise = torch.zeros(len(P))
        if tree.is_root() and params.dirichlet_mult:
            noise = Dirichlet([params.dirichlet_alpha] * len(P)).sample()
            P = (1.0 - params.dirichlet_mult) * P + params.dirichlet_mult * noise

        V = torch.Tensor([c.effective_value_avg(current_player) for c in tree.children])
        N = torch.Tensor([c.effective_count() for c in tree.children])
        eps = 1e-6  # needed when N == 0
        PUCT = V + c_PUCT * P * (np.sqrt(sum(N) + eps)) / (1 + N)
        value_avg = torch.Tensor([tree.effective_value_avg(p) for p in range(tree.n_players)])
        E = torch.Tensor([c.eliminated for c in tree.children])
        PUCT *= 1 - E

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
            player=current_player, depth=depth, board=board, evaluation=evaluation, leaf=leaf, value_avg=value_avg,
            tree=tree, debug_subtree=debug_subtree, P=P, noise=noise, V=V, N=N, E=E, PUCT=PUCT)


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
