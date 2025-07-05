from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import Agent, MCTSAgent, AgentRole, IndexedAgent
from alphazero.logic.custom_types import ClientConnection, ClientId, Domain, FileToTransfer, \
    Generation, ServerStatus
from alphazero.logic.evaluator import EvalUtils
from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import estimate_elo_newton, WinLossDrawCounts
from alphazero.logic.rating_db import DBAgentRating, RatingDB
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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union


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
    result: Optional[WinLossDrawCounts] = None


@dataclass
class EvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: Dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    status: Optional[EvalRequestStatus] = None
    elo: Optional[float] = None


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

        self._indexed_agents: List[IndexedAgent] = []
        self._agent_lookup: Dict[Agent, IndexedAgent] = {}
        self._agent_lookup_db_id: Dict[int, IndexedAgent] = {}
        self._eval_status_dict: Dict[int, EvalStatus] = {} # ix -> EvalStatus
        self._benchmark_elos: Dict[int, float] = {}  # ix -> elo
        self._min_benchmark_elo: Optional[float] = None
        self._max_benchmark_elo: Optional[float] = None
        self._db = RatingDB(self._controller._organizer.eval_db_filename(benchmark_tag))

    def set_priority(self):
        dict_len = len([ix for ix, data in self._eval_status_dict.items() if data.status == EvalRequestStatus.COMPLETE])
        rating_in_progress = any(data.status == EvalRequestStatus.REQUESTED for data in self._eval_status_dict.values())
        self._set_domain_priority(dict_len, rating_in_progress)

    def load_past_data(self):
        self._load_agents_from_db()
        self._load_matches_from_db()
        self._load_elos_from_db()
        self._load_benchmark_elos()
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
        assert conn.is_on_localhost()
        eval_ix = conn.aux.ix
        if eval_ix is None:
            gen = self._get_next_gen_to_eval()
            assert gen is not None
            test_agent = MCTSAgent(gen, n_iters=self.n_iters, set_temp_zero=True, tag=self._controller._organizer.tag)
            test_iagent = self._add_agent(test_agent, {AgentRole.TEST}, db=self._db)

            conn.aux.ix = test_iagent.index
            with self._lock:
                if test_iagent.index in self._eval_status_dict:
                    assert self._eval_status_dict[test_iagent.index].status == EvalRequestStatus.FAILED
                    self._eval_status_dict[test_iagent.index].status = EvalRequestStatus.REQUESTED
                else:
                    self._eval_status_dict[test_iagent.index] = EvalStatus(mcts_gen=gen, owner=conn.client_id,
                                                                           ix_match_status={},
                                                                           status=EvalRequestStatus.REQUESTED)
            conn.aux.needs_new_opponents = True
            self.set_priority()
        else:
            test_iagent = self._indexed_agents[eval_ix]

        estimated_rating = conn.aux.estimated_rating
        if estimated_rating is None:
            estimated_rating = self._estimate_rating(test_iagent)
            logger.info('Estimated rating for gen %s: %s', test_iagent.agent.gen, estimated_rating)
            conn.aux.estimated_rating = estimated_rating

        n_games_completed = sum([data.n_games for data in self._eval_status_dict[test_iagent.index].ix_match_status.values() \
            if data.status == MatchRequestStatus.COMPLETE])
        n_games_to_do = self.n_games - n_games_completed

        opponent_ixs_played = [ix for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() \
            if data.status in (MatchRequestStatus.COMPLETE, MatchRequestStatus.REQUESTED)]

        if n_games_to_do <= 0 or len(opponent_ixs_played) >= len(self._benchmark_elos):
            self._calc_ratings(conn, conn.aux.ix)
            self._set_ready(conn)
            return

        n_games_in_progress = sum([data.n_games for data in self._eval_status_dict[test_iagent.index].ix_match_status.values() \
            if data.status == MatchRequestStatus.REQUESTED])
        n_games_needed = n_games_to_do - n_games_in_progress
        assert n_games_needed > 0, f"{self.n_games} games needed, but {n_games_in_progress} already in progress"

        need_new_opponents = conn.aux.needs_new_opponents
        if need_new_opponents:
            logger.debug('Requesting %s games for gen %s, estimated rating: %s', n_games_needed, test_iagent.agent.gen, estimated_rating)
            chosen_ixs, num_matches = self._gen_matches(test_iagent.index, estimated_rating, n_games_needed)
            logger.debug('chosen ixs and num matches: %s', list(zip(chosen_ixs, num_matches)))
            with self._lock:
                self._update_eval_status(test_iagent.index, chosen_ixs, num_matches)
            conn.aux.needs_new_opponents = False

        candidates = [(ix, data.n_games) for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() if data.status == MatchRequestStatus.PENDING]
        next_opponent_ix, next_n_games = sorted(candidates, key=lambda x: x[1])[0]

        next_opponent_iagent = self._indexed_agents[next_opponent_ix]

        data = self._gen_match_request_data(test_iagent, next_opponent_iagent, next_n_games)
        conn.socket.send_json(data)
        self._eval_status_dict[test_iagent.index].ix_match_status[next_opponent_ix].status = MatchRequestStatus.REQUESTED

    def _load_agents_from_db(self):
        for db_agent in self._db.fetch_agents():
            indexed_agent = IndexedAgent(agent=db_agent.agent,
                                         index=len(self._indexed_agents),
                                         roles=db_agent.roles,
                                         db_id=db_agent.db_id)
            self._indexed_agents.append(indexed_agent)
            self._agent_lookup[indexed_agent.agent] = indexed_agent
            self._agent_lookup_db_id[indexed_agent.db_id] = indexed_agent

    def _load_matches_from_db(self):
        for result in self._db.fetch_match_results():
            indexed_agent1 = self._agent_lookup_db_id[result.agent_id1]
            indexed_agent2 = self._agent_lookup_db_id[result.agent_id2]
            assert AgentRole.TEST in indexed_agent1.roles

            if indexed_agent1.index not in self._eval_status_dict:
                self._eval_status_dict[indexed_agent1.index] = EvalStatus(
                    mcts_gen=indexed_agent1.agent.gen,
                    owner=None,
                    ix_match_status={},
                    status=EvalRequestStatus.FAILED
                )

            match_status = MatchStatus(opponent_level=indexed_agent2.agent.level,
                                       n_games=result.counts.n_games,
                                       status=MatchRequestStatus.COMPLETE,
                                       result=result.counts)
            self._eval_status_dict[indexed_agent1.index].ix_match_status[indexed_agent2.index] = match_status

    def _load_benchmark_elos(self):
        ratings: List[DBAgentRating] = self._db.load_ratings(AgentRole.BENCHMARK)
        for db_rating in ratings:
            if db_rating.is_committee:
                indexed_agent = self._agent_lookup_db_id[db_rating.agent_id]
                self._benchmark_elos[indexed_agent.index] = db_rating.rating
        elos = np.array(list(self._benchmark_elos.values()))
        self._min_benchmark_elo = elos.min()
        self._max_benchmark_elo = elos.max()

    def _load_elos_from_db(self):
        ratings: List[DBAgentRating] = self._db.load_ratings(AgentRole.TEST)
        for db_rating in ratings:
            indexed_agent = self._agent_lookup_db_id[db_rating.agent_id]
            self._eval_status_dict[indexed_agent.index].elo = db_rating.rating
            self._eval_status_dict[indexed_agent.index].status = EvalRequestStatus.COMPLETE

    def _add_agent(self, agent: Agent, roles: set[AgentRole], db: RatingDB) -> IndexedAgent:
        iagent = self._agent_lookup.get(agent, None)
        if iagent is not None:
            return iagent

        index = len(self._indexed_agents)
        iagent = IndexedAgent(agent=agent, index=index, roles=roles, db_id=None)
        self._indexed_agents.append(iagent)
        self._agent_lookup[agent] = iagent
        db.commit_agent(iagent)
        self._agent_lookup_db_id[iagent.db_id] = iagent
        return iagent

    def _gen_matches(self, ix: int, estimated_rating: float, n_games_needed: int) -> Tuple[np.ndarray, np.ndarray]:
        ixs = []
        elos = []
        ix_match_status: Dict[int, MatchStatus] = self._eval_status_dict[ix].ix_match_status
        for benchmark_ix, benchmark_elo in self._benchmark_elos.items():
            if benchmark_ix in ix_match_status and ix_match_status[benchmark_ix].status in (MatchRequestStatus.REQUESTED, MatchRequestStatus.COMPLETE):
                continue
            ixs.append(benchmark_ix)
            elos.append(benchmark_elo)
        return EvalUtils.gen_matches(estimated_rating, ixs, elos, n_games_needed)

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
        if any(match_status.status == MatchRequestStatus.COMPLETE for match_status in self._eval_status_dict[test_iagent.index].ix_match_status.values()):
            estimated_rating = self._eval_elo(test_iagent.index)
        else:
            estimated_rating = self._estimate_rating_nearby_gens(test_iagent.agent.gen)
            if estimated_rating is None:
                estimated_rating = self._min_benchmark_elo
        return estimated_rating

    def _eval_elo(self, ix: int) -> float:
        eval_status: EvalStatus = self._eval_status_dict[ix]
        n = []
        k = []
        elos = []
        for ix, match_status in eval_status.ix_match_status.items():
            if match_status.status == MatchRequestStatus.COMPLETE:
                n.append(match_status.n_games)
                k.append(match_status.result.win + 0.5 * match_status.result.draw)
                elos.append(self._benchmark_elos[ix])
        assert len(n) > 0
        return estimate_elo_newton(np.array(n), np.array(k), np.array(elos), lower=self._min_benchmark_elo, upper=self._max_benchmark_elo)

    def _estimate_rating_nearby_gens(self, gen):
        evaluated_gens = []
        elos = []
        for eval_status in self._eval_status_dict.values():
            if eval_status.status == EvalRequestStatus.COMPLETE:
                evaluated_gens.append(eval_status.mcts_gen)
                elos.append(eval_status.elo)

        if not evaluated_gens:
            return None
        estimated_rating = EvalUtils.estimate_rating_nearby_gens(gen, evaluated_gens, elos)
        return estimated_rating

    def _update_eval_status(self, test_ix, chosen_ixs, num_matches):
        for ix, match_status in self._eval_status_dict[test_ix].ix_match_status.items():
            if ix not in chosen_ixs and match_status.status == MatchRequestStatus.PENDING:
                match_status.status = MatchRequestStatus.OBSOLETE
                logger.debug('...set match between %s and %s to obsolete', self._indexed_agents[test_ix].index, match_status.opponent_level)

        for ix, n_games in zip(chosen_ixs, num_matches):
            if ix not in self._eval_status_dict[test_ix].ix_match_status:
                agent: Agent = self._indexed_agents[ix].agent
                new_opponent_level = agent.level
                self._eval_status_dict[test_ix].ix_match_status[ix] = \
                    MatchStatus(new_opponent_level, n_games, MatchRequestStatus.PENDING)
                logger.debug('+++add new %s matches between %s and %s', n_games, self._indexed_agents[test_ix].index, ix)
            elif self._eval_status_dict[test_ix].ix_match_status[ix].status == MatchRequestStatus.REQUESTED:
                continue
            else:
                assert self._eval_status_dict[test_ix].ix_match_status[ix].status in (MatchRequestStatus.PENDING, MatchRequestStatus.OBSOLETE)
                self._eval_status_dict[test_ix].ix_match_status[ix].n_games = n_games
                self._eval_status_dict[test_ix].ix_match_status[ix].status = MatchRequestStatus.PENDING
                logger.debug('+++modify to %s matches between %s and %s', n_games, self._indexed_agents[test_ix].index, ix)

    def handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        logger.debug('---Received match result for ix1=%s, ix2=%s, counts=%s', ix1, ix2, counts)

        with self._db.db_lock:
            self._update_match_results(ix1, ix2, counts)

        new_rating = self._eval_elo(ix1)
        old_rating = conn.aux.estimated_rating
        if abs(new_rating - old_rating) > self.error_threshold:
            logger.debug('Rating change too large, old=%s, new=%s. requesting new opponents...', old_rating, new_rating)
            conn.aux.needs_new_opponents = True
            conn.aux.estimated_rating = new_rating

        with self._lock:
            self._eval_status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.status == MatchRequestStatus.PENDING for v in self._eval_status_dict[ix1].ix_match_status.values())

        logger.debug('Has pending matches for ix %s: %s', ix1, has_pending)
        if not has_pending:
            self._calc_ratings(conn, ix1, new_rating)

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        logger.debug('Inside handle_match_results, releasing GPU lock for %s after match result', conn.client_domain)
        table.release_lock(conn.client_domain)
        self.set_priority()
        logger.debug('End of handle_match_result')

    def _update_match_results(self, ix1: int, ix2: int, counts: WinLossDrawCounts):
        db_id1 = self._indexed_agents[ix1].db_id
        db_id2 = self._indexed_agents[ix2].db_id
        self._db.commit_counts(db_id1, db_id2, counts, MatchType.EVALUATE)
        self._eval_status_dict[ix1].ix_match_status[ix2].result = counts
        self._eval_status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE

    def _calc_ratings(self, conn: ClientConnection, eval_ix: int, rating: Optional[float]=None):
        self._eval_status_dict[eval_ix].status = EvalRequestStatus.COMPLETE
        self._eval_status_dict[eval_ix].owner = None
        ix = conn.aux.ix
        assert ix == eval_ix
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        logger.debug('Inside _calc_ratings, releasing GPU lock for %s', conn.client_domain)
        table.release_lock(conn.client_domain)

        if rating is None:
            logger.debug('Calculating rating for gen %s...', self._indexed_agents[eval_ix].agent)
            rating = self._eval_elo(eval_ix)
            logger.debug('Calculated rating for gen %s: %s', self._indexed_agents[eval_ix].agent, rating)

        test_iagent = self._indexed_agents[eval_ix]
        with self._db.db_lock:
            self._db.commit_ratings([test_iagent], [rating])
        self._eval_status_dict[eval_ix].elo = rating
        conn.aux.estimated_rating = None
        conn.aux.ix = None
        logger.debug('///Finished evaluating gen %s, rating: %s', test_iagent.agent, rating)

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

