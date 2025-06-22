from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import Agent, MCTSAgent, AgentRole, IndexedAgent
from alphazero.logic.custom_types import ClientConnection, ClientId, Domain, FileToTransfer, \
    Generation, ServerStatus
from alphazero.logic.evaluator import Evaluator, EvalUtils
from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.gaming_manager_base import GamingManagerBase, ManagerConfig, \
    ServerAuxBase, WorkerAux
from util.socket_util import JsonDict

import numpy as np

from dataclasses import dataclass, field, replace
from enum import Enum
import logging
import os
import threading
from typing import Dict, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)
Strength = int


class MatchRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    PENDING = 'pending'
    OBSOLETE = 'obsolete'


class EvalRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    FAILED = 'failed'


@dataclass
class MatchStatus:
    opponent_level: Union[Strength, Generation]
    n_games: int
    status: MatchRequestStatus


@dataclass
class EvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: Dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    status: Optional[EvalRequestStatus] = None


@dataclass
class EvalServerAux(ServerAuxBase):
    """
    Auxiliary data stored per server connection.
    """
    needs_new_opponents: bool = True
    estimated_rating: Optional[float] = None
    ix: Optional[int] = None  # test agent index

    def work_in_progress(self) -> bool:
        return self.ix is not None

class EvalManager(GamingManagerBase):
    """
    A separate EvalManager is created for each rating-tag.
    """
    def __init__(self, controller: LoopController, benchmark_tag: str):
        manager_config = ManagerConfig(
            worker_aux_class=WorkerAux,
            server_aux_class=EvalServerAux,
            server_name='eval-server',
            worker_name='eval-worker',
            domain=Domain.EVAL,
        )
        super().__init__(controller, manager_config)
        self._evaluator = Evaluator(self._controller._organizer, benchmark_tag)
        self._eval_status_dict: Dict[int, EvalStatus] = {} # ix -> EvalStatus

    def set_priority(self):
        dict_len = len(self._eval_status_dict)
        rating_in_progress = any(data.status == EvalRequestStatus.REQUESTED for data in self._eval_status_dict.values())
        self._set_domain_priority(dict_len, rating_in_progress)

    def load_past_data(self):
        logger.info('Loading past ratings data...')
        rating_data = self._evaluator.read_ratings_from_db()
        evaluated_ixs = [iagent.index for iagent in rating_data.evaluated_iagents]
        test_ixs = self._evaluator.test_agent_ixs()
        for test_ix in test_ixs:
            gen = self._evaluator.indexed_agents[test_ix].agent.gen
            ix_match_status = self._load_ix_match_status(test_ix)
            self._eval_status_dict[test_ix] = EvalStatus(mcts_gen=gen, ix_match_status=ix_match_status)
            if test_ix not in evaluated_ixs:
                self._eval_status_dict[test_ix].status = EvalRequestStatus.FAILED
            else:
                self._eval_status_dict[test_ix].status = EvalRequestStatus.COMPLETE

        self.set_priority()

    def num_evaluated_gens(self):
        return sum(1 for data in self._eval_status_dict.values() if data.status == EvalRequestStatus.COMPLETE)

    def handle_server_disconnect(self, conn: ClientConnection):
        logger.debug('Server disconnected: %s, evaluating ix %s', conn, conn.aux.ix)
        ix = conn.aux.ix
        if ix is not None:
            with self._lock:
                eval_status = self._eval_status_dict[ix].status
                if eval_status != EvalRequestStatus.COMPLETE:
                    self._eval_status_dict[ix].status = EvalRequestStatus.FAILED
                    self._eval_status_dict[ix].owner = None
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            conn.aux.status= ServerStatus.DISCONNECTED
            status_cond.notify_all()

    def send_match_request(self, conn: ClientConnection):
        self._evaluator.refresh_ratings()
        assert conn.is_on_localhost()
        eval_ix = conn.aux.ix
        if eval_ix is None:
            gen = self._get_next_gen_to_eval()
            assert gen is not None
            test_agent = MCTSAgent(gen, n_iters=self.n_iters, set_temp_zero=True, tag=self._controller._organizer.tag)
            test_iagent = self._evaluator.add_agent(test_agent, {AgentRole.TEST}, expand_matrix=True, db=self._evaluator.db)
            conn.aux.ix = test_iagent.index
            with self._lock:
                if test_iagent.index in self._eval_status_dict:
                    assert self._eval_status_dict[test_iagent.index].status == EvalRequestStatus.FAILED
                    self._eval_status_dict[test_iagent.index].status = EvalRequestStatus.REQUESTED
                else:
                    ix_match_status = self._load_ix_match_status(test_iagent.index)
                    self._eval_status_dict[test_iagent.index] = EvalStatus(mcts_gen=gen, owner=conn.client_id,
                                                                           ix_match_status=ix_match_status,
                                                                           status=EvalRequestStatus.REQUESTED)
            conn.aux.needs_new_opponents = True
            self.set_priority()
        else:
            test_iagent = self._evaluator.indexed_agents[eval_ix]

        estimated_rating = conn.aux.estimated_rating
        if estimated_rating is None:
            estimated_rating = self._estimate_rating(test_iagent)
            logger.debug('Estimated rating for gen %s: %s', test_iagent.agent.gen, estimated_rating)
            conn.aux.estimated_rating = estimated_rating

        n_games_completed = sum([data.n_games for data in self._eval_status_dict[test_iagent.index].ix_match_status.values() \
            if data.status == MatchRequestStatus.COMPLETE])
        n_games_to_do = self.n_games - n_games_completed

        opponent_ixs_played = [ix for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() \
            if data.status in (MatchRequestStatus.COMPLETE, MatchRequestStatus.REQUESTED)]

        if n_games_to_do <= 0 or len(opponent_ixs_played) >= len(self._evaluator.benchmark_committee):
            self._interpolate_ratings(conn, conn.aux.ix)
            self._set_ready(conn)
            return

        n_games_in_progress = sum([data.n_games for data in self._eval_status_dict[test_iagent.index].ix_match_status.values() \
            if data.status == MatchRequestStatus.REQUESTED])
        n_games_needed = n_games_to_do - n_games_in_progress
        assert n_games_needed > 0, f"{self.n_games} games needed, but {n_games_in_progress} already in progress"

        need_new_opponents = conn.aux.needs_new_opponents
        if need_new_opponents:
            logger.debug('Requesting %s games for gen %s, estimated rating: %s', n_games_needed, test_iagent.agent.gen, estimated_rating)
            chosen_ixs, num_matches = self._evaluator.gen_matches(estimated_rating, opponent_ixs_played, n_games_needed)
            logger.debug('chosen ixs and num matches: %s', list(zip(chosen_ixs, num_matches)))
            with self._lock:
                self._update_eval_status(test_iagent.index, chosen_ixs, num_matches)
            conn.aux.needs_new_opponents = False

        candidates = [(ix, data.n_games) for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() if data.status == MatchRequestStatus.PENDING]
        next_opponent_ix, next_n_games = sorted(candidates, key=lambda x: x[1])[0]
        next_opponent_iagent = self._evaluator.indexed_agents[next_opponent_ix]

        data = self._gen_match_request_data(test_iagent, next_opponent_iagent, next_n_games)
        conn.socket.send_json(data)
        self._eval_status_dict[test_iagent.index].ix_match_status[next_opponent_ix].status = MatchRequestStatus.REQUESTED

    def _gen_match_request_data(self, test_iagent: IndexedAgent, next_opponent_iagent: IndexedAgent, next_n_games) -> JsonDict:
        game = self._controller._run_params.game
        next_opponent_agent = next_opponent_iagent.agent

        benchmark_organizer = None
        if next_opponent_agent.tag:
            benchmark_organizer = DirectoryOrganizer(RunParams(game, next_opponent_agent.tag), base_dir_root='/workspace')

        eval_binary_src = self._controller._get_binary_path()
        benchmark_binary_src = self._controller._get_binary_path(benchmark_organizer=benchmark_organizer)

        eval_binary = FileToTransfer.from_src_scratch_path(
            source_path=eval_binary_src,
            scratch_path=f'bin/{game}',
            asset_path_mode='hash'
        )

        benchmark_binary = FileToTransfer.from_src_scratch_path(
            source_path=benchmark_binary_src,
            scratch_path=f'benchmark-bin/{game}',
            asset_path_mode='hash'
        )
        files_required = [eval_binary, benchmark_binary]

        eval_model = None
        if test_iagent.agent.gen > 0:
            eval_model = FileToTransfer.from_src_scratch_path(
                source_path=self._controller._organizer.get_model_filename(test_iagent.agent.gen),
                scratch_path=f'eval-models/{test_iagent.agent.tag}/gen-{test_iagent.agent.gen}.pt',
                asset_path_mode='scratch'
            )
            files_required.append(eval_model)

        if isinstance(next_opponent_agent, MCTSAgent):
            benchmark_organizer = DirectoryOrganizer(RunParams(game, next_opponent_agent.tag), base_dir_root='/workspace')
            benchmark_binary = FileToTransfer.from_src_scratch_path(
                source_path=benchmark_organizer.binary_filename,
                scratch_path=f'benchmark-bin/{game}',
                asset_path_mode='hash'
            )
            files_required.append(benchmark_binary)

            benchmark_model = None
            if next_opponent_agent.gen > 0:
                benchmark_model = FileToTransfer.from_src_scratch_path(
                    source_path=benchmark_organizer.get_model_filename(next_opponent_agent.gen),
                    scratch_path=f'benchmark-models/{next_opponent_agent.tag}/gen-{next_opponent_agent.gen}.pt',
                    asset_path_mode='scratch'
                )
                files_required.append(benchmark_model)

            opponent_agent = replace(next_opponent_agent,
                                     binary=benchmark_binary.scratch_path,
                                     model=benchmark_model.scratch_path if benchmark_model else None)
        else:
            opponent_agent = replace(next_opponent_agent, binary=eval_binary.scratch_path)
            for dep in self._controller.game_spec.extra_runtime_deps:
                dep_file = FileToTransfer.from_src_scratch_path(
                    source_path=os.path.join('/workspace/repo/', dep),
                    scratch_path=dep,
                    asset_path_mode='hash'
                )
                files_required.append(dep_file)

        data = {
            'type': 'match-request',
            'agent1': {
                'type': 'MCTS',
                'data': {
                    'gen': test_iagent.agent.gen,
                    'n_iters': self.n_iters,
                    'set_temp_zero': True,
                    'tag': self._controller._organizer.tag,
                    'binary': eval_binary.scratch_path,
                    'model': eval_model.scratch_path if eval_model else None,
                    }
                },
            'agent2': {
                'type':'MCTS' if isinstance(next_opponent_agent, MCTSAgent) else 'Reference',
                'data': opponent_agent.to_dict()
                },
            'ix1': test_iagent.index,
            'ix2': int(next_opponent_iagent.index),
            'n_games': int(next_n_games),
            'files_required': [f.to_dict() for f in files_required],
        }

        n_games = data['n_games']
        logger.info(f"Evaluating {test_iagent.agent} vs {next_opponent_agent}, ({n_games} games)")
        return data

    def _get_next_gen_to_eval(self):
        failed_gen = [data.mcts_gen for data in self._eval_status_dict.values() \
            if data.status == EvalRequestStatus.FAILED]
        if failed_gen:
            return failed_gen[0]

        latest_gen = self._controller._organizer.get_latest_model_generation()
        evaluated_gens = [data.mcts_gen for data in self._eval_status_dict.values() \
            if data.status in (EvalRequestStatus.COMPLETE, EvalRequestStatus.REQUESTED)]
        gen = EvalUtils.get_next_gen_to_eval(latest_gen, evaluated_gens)
        return gen

    def _estimate_rating(self, test_iagent):
        if self._evaluator._arena.n_games_played(test_iagent.agent) > 0:
            estimated_rating = self._evaluator.arena_ratings[test_iagent.index]
        else:
            estimated_rating = self._estimate_rating_nearby_gens(test_iagent.agent.gen)
            if estimated_rating is None:
                estimated_rating = np.mean(self._evaluator.arena_ratings)
        assert estimated_rating is not None
        return estimated_rating

    def _estimate_rating_nearby_gens(self, gen):
        evaluated_data = [(ix, data.mcts_gen) for ix, data in \
            self._eval_status_dict.items() if data.status == EvalRequestStatus.COMPLETE]
        if not evaluated_data:
            return None
        evaluated_ixs, evaluated_gens = zip(*evaluated_data)
        ratings = self._evaluator.arena_ratings[np.array(evaluated_ixs)]
        estimated_rating = EvalUtils.estimate_rating_nearby_gens(gen, evaluated_gens, ratings)
        return estimated_rating

    def _update_eval_status(self, test_ix, chosen_ixs, num_matches):
        for ix, match_status in self._eval_status_dict[test_ix].ix_match_status.items():
            if ix not in chosen_ixs and match_status.status == MatchRequestStatus.PENDING:
                match_status.status = MatchRequestStatus.OBSOLETE
                logger.debug('...set match between %s and %s to obsolete', self._evaluator.indexed_agents[test_ix].index, match_status.opponent_level)

        for ix, n_games in zip(chosen_ixs, num_matches):
            if ix not in self._eval_status_dict[test_ix].ix_match_status:
                agent: Agent = self._evaluator.indexed_agents[ix].agent
                new_opponent_level = agent.level
                self._eval_status_dict[test_ix].ix_match_status[ix] = \
                    MatchStatus(new_opponent_level, n_games, MatchRequestStatus.PENDING)
                logger.debug('+++add new %s matches between %s and %s', n_games, self._evaluator.indexed_agents[test_ix].index, ix)
            else:
                self._eval_status_dict[test_ix].ix_match_status[ix].n_games = n_games
                if self._eval_status_dict[test_ix].ix_match_status[ix].status == MatchRequestStatus.OBSOLETE:
                    self._eval_status_dict[test_ix].ix_match_status[ix].status = MatchRequestStatus.PENDING
                logger.debug('+++modify to %s matches between %s and %s', n_games, self._evaluator.indexed_agents[test_ix].index, ix)

    def _load_ix_match_status(self, test_ix: int) -> Dict[int, MatchStatus]:
        W_matrix = self._evaluator._arena._W_matrix
        n_games_played = W_matrix[test_ix, :] + W_matrix[:, test_ix]
        match_status_dict = {}
        for ix, n_games in enumerate(n_games_played):
            if ix == test_ix:
                continue
            if n_games > 0:
                level = self._evaluator.indexed_agents[ix].agent.level
                match_status_dict[ix] = MatchStatus(level, int(n_games), MatchRequestStatus.COMPLETE)
        return match_status_dict

    def handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        logger.debug('---Received match result for ix1=%s, ix2=%s, counts=%s', ix1, ix2, counts)
        with self._evaluator.db.db_lock:
            self._evaluator._arena.update_match_results(ix1, ix2, counts, MatchType.EVALUATE, self._evaluator.db)
        self._evaluator.refresh_ratings()
        new_rating = self._evaluator.arena_ratings[ix1]
        old_rating = conn.aux.estimated_rating
        logger.debug('Old rating: %s, New rating: %s', old_rating, new_rating)
        if abs(new_rating - old_rating) > self.error_threshold:
            logger.debug('Rating change too large, requesting new opponents...')
            conn.aux.needs_new_opponents = True
            conn.aux.estimated_rating = new_rating

        with self._lock:
            self._eval_status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.status == MatchRequestStatus.PENDING for v in self._eval_status_dict[ix1].ix_match_status.values())

        if not has_pending:
            self._interpolate_ratings(conn, ix1)

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.release_lock(conn.client_domain)
        self.set_priority()

    def _interpolate_ratings(self, conn: ClientConnection, eval_ix: int):
        # assert self._evaluator._arena.n_games_played(self._evaluator.indexed_agents[eval_ix].agent) == self.n_games
        self._eval_status_dict[eval_ix].status = EvalRequestStatus.COMPLETE
        self._eval_status_dict[eval_ix].owner = None
        ix = conn.aux.ix
        assert ix == eval_ix
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.release_lock(conn.client_domain)

        test_ixs, interpolated_ratings = self._evaluator.interpolate_ratings()
        test_iagents = [self._evaluator.indexed_agents[ix] for ix in test_ixs]
        with self._evaluator.db.db_lock:
            self._evaluator.db.commit_ratings(test_iagents, interpolated_ratings)
        conn.aux.estimated_rating = None
        conn.aux.ix = None
        logger.debug('///Finished evaluating gen %s, rating: %s',
                    self._evaluator.indexed_agents[eval_ix].agent.gen,
                    interpolated_ratings[np.where(test_ixs == eval_ix)[0]])

    def _task_finished(self) -> None:
        rated_percent = self.num_evaluated_gens() / self._controller._organizer.get_latest_model_generation()
        return rated_percent >= self._controller.params.target_rating_rate

    @property
    def n_games(self):
        return self._controller.rating_params.n_games_per_evaluation

    @property
    def n_iters(self):
        return self._controller.rating_params.rating_player_options.num_iterations

    @property
    def error_threshold(self):
        return self._controller.rating_params.eval_error_threshold
