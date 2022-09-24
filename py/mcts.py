"""
Action items:

1. MCGS instead of MCTS (Tree -> Graph)? This makes sense because different ordering of moves can arrive at the same
   game state. But, the neural net takes recent history as input, which means the states aren't identical from the
   net's point of view. How do we deal with that? How do we deal with graph cycles?

2. Neural net cache.

3. Subtree reuse (how does this interact with Dirichlet noise?).

4. Root parallelism. In turn demands multiprocess setup, virtual loss, etc.
"""
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch import Tensor

from train import AbstractGameState, AbstractNeuralNetwork, ActionIndex, ActionMask, PlayerIndex, \
    GlobalPolicyLogitDistr, LocalPolicyProbDistr, ValueProbDistr, GlobalPolicyProbDistr

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
    def __init__(self, network: AbstractNeuralNetwork, state: AbstractGameState):
        self.current_player: PlayerIndex = state.getCurrentPlayer()
        self.game_result: Optional[ValueProbDistr] = state.getGameResult()

        self.valid_action_mask: Optional[ActionMask] = None
        self.policy_logit_distr: Optional[GlobalPolicyLogitDistr] = None
        self.value_prob_distr: Optional[ValueProbDistr] = None

        if self.game_result is not None:
            # game is over, don't bother computing other fields
            return

        self.valid_action_mask = state.getValidActions()
        policy_output, value_output = network.evaluate(state.vectorize())
        self.policy_logit_distr = policy_output
        self.value_prob_distr = value_output.softmax(dim=0)


class Tree:
    def __init__(self, n_players: int, action_index: Optional[ActionIndex] = None, parent: Optional['Tree'] = None):
        self.state: Optional[AbstractGameState] = None  # use only when relying on state-clones
        self.evaluation: Optional[StateEvaluation] = None
        self.valid_action_indices: Optional[List[ActionIndex]] = None
        self.children: Optional[List[Tree]] = None
        self.action_index: Optional[ActionIndex] = action_index
        self.parent = parent
        self.count = 0
        self.value_sum = torch.zeros(n_players)
        self.value_prior: Optional[ValueProbDistr] = None
        self.policy_prior: Optional[LocalPolicyProbDistr] = None

    def store_state(self, state: AbstractGameState):
        if self.state is not None:
            self.state = state.clone()

    @property
    def n_players(self) -> int:
        return self.value_sum.shape[0]

    def is_root(self) -> bool:
        return self.parent is None

    def win_rates(self) -> ValueProbDistr:
        return self.value_sum / (self.count if self.count else 1.0)

    def avg_value(self, player_index: PlayerIndex):
        if self.count == 0:
            return 0.0
        return self.value_sum[player_index] / self.count

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
        if self.parent:
            self.parent.backprop(result)

    def compute_policy_prior(self, evaluation: StateEvaluation, params: MCTSParams) -> LocalPolicyProbDistr:
        if self.policy_prior is not None:
            return self.policy_prior

        valid_action_mask = evaluation.valid_action_mask
        policy_output = evaluation.policy_logit_distr

        is_root = self.is_root()
        inv_temp = (1.0 / params.root_softmax_temperature) if is_root else 1.0

        valid_action_indices = torch.where(valid_action_mask)[0]
        P = torch.softmax(policy_output[valid_action_indices] * inv_temp, dim=0)
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
    def __init__(self, network: AbstractNeuralNetwork, n_players: int = 2, debug_file: Optional[io.StringIO] = None):
        self.debug_file = debug_file
        self.network = network
        self.n_players = n_players
        self.root: Optional[Tree] = None
        self.cache: Dict[Any, StateEvaluation] = {}

    def sim(self, state: AbstractGameState, params: MCTSParams) -> MCTSResults:
        self.root = Tree(self.n_players)
        for i in range(params.treeSizeLimit):
            if self.debug_file:
                self.debug_file.write('****************************\n')
                self.debug_file.write(f'SIM LOOP i={i}\n')
                self.debug_file.write('****************************\n')
                self.debug_file.flush()
            self.visit(self.root, state, params)

        n = state.getNumGlobalActions()
        counts = torch.zeros(n, dtype=int)
        for child in self.root.children:
            counts[child.action_index] = child.count
        win_rates = self.root.win_rates()
        policy_prior = torch.zeros(n)
        policy_prior[self.root.valid_action_indices] = self.root.policy_prior
        value_prior = self.root.value_prior
        return MCTSResults(counts, win_rates, policy_prior, value_prior)

    def evaluate(self, tree: Tree, state: AbstractGameState) -> StateEvaluation:
        if tree.evaluation is not None:
            return tree.evaluation

        signature = state.getSignature()
        evaluation = self.cache.get(signature, None)
        if evaluation is None:
            evaluation = StateEvaluation(self.network, state)
            self.cache[signature] = evaluation

        tree.evaluation = evaluation
        return evaluation

    def visit(self, tree: Tree, state: AbstractGameState, params: MCTSParams):
        evaluation = self.evaluate(tree, state)
        game_result = evaluation.game_result

        if game_result is not None:
            tree.backprop(game_result)
            return

        current_player = evaluation.current_player
        leaf = not tree.has_children()
        tree.expand_children(evaluation)

        c_PUCT = params.c_PUCT
        P = tree.compute_policy_prior(evaluation, params)
        if tree.is_root() and params.dirichlet_mult:
            noise = np.random.dirichlet([params.dirichlet_alpha] * len(P))
            P = (1.0 - params.dirichlet_mult) * P + params.dirichlet_mult * noise

        V = torch.Tensor([c.avg_value(current_player) for c in tree.children])
        N = torch.Tensor([c.count for c in tree.children])
        eps = 1e-6  # needed when N == 0
        PUCT = V + c_PUCT * P * (np.sqrt(sum(N) + eps)) / (1 + N)

        best_child = tree.children[np.argmax(PUCT)]

        if self.debug_file:
            state.debugDump(self.debug_file)
            self.debug_file.write(f'*** visit action:{tree.action_index}\n')
            self.debug_file.write(f'P: {P}\n')
            self.debug_file.write(f'V: {V}\n')
            self.debug_file.write(f'N: {N}\n')
            self.debug_file.write(f'PUCT: {PUCT}\n')
            self.debug_file.write(f'sum: {tree.value_sum}\n')
            self.debug_file.flush()

        if leaf:
            tree.backprop(evaluation.value_prob_distr)
            if self.debug_file:
                self.debug_file.write(f'BACKPROP: {evaluation.value_prob_distr}\n')
                self.debug_file.write(f'root sum: {self.root.value_sum}')
                self.debug_file.flush()
        else:
            if state.supportsUndo():
                state.applyMove(best_child.action_index)
                self.visit(best_child, state, params)
                state.undoLastMove()
            else:
                best_child.store_state(state)
                state.applyMove(best_child.action_index)
                self.visit(best_child, best_child.state, params)
