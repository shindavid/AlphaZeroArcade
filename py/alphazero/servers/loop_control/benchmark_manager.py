from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import MCTSAgent, AgentRole, IndexedAgent, Match, MatchType
from alphazero.logic.benchmarker import Benchmarker, BenchmarkRatingData
from alphazero.logic.custom_types import ClientConnection, ClientId, EvalTag, FileToTransfer, \
    Generation, ServerStatus

from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.base_manager import BaseManager, ManagerConstants
from util.socket_util import JsonDict

import numpy as np

from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from typing import List, Optional


logger = logging.getLogger(__name__)


class MatchRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    PENDING = 'pending'


class BenchmarkRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    FAILED = 'failed'


@dataclass
class MatchStatus:
    opponent_gen: Generation
    n_games: int
    status: MatchRequestStatus


@dataclass
class BenchmarkStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    status: Optional[BenchmarkRequestStatus] = None


@dataclass
class ServerAux:
    """
    Auxiliary data stored per server connection.
    """
    status_cond: threading.Condition = field(default_factory=threading.Condition)
    status: ServerStatus = ServerStatus.BLOCKED
    ix: Optional[int] = None  # test agent index
    ready_for_latest_gen: bool = False

    def work_in_progress(self) -> bool:
        return self.ix is not None

class BenchmarkManager(BaseManager):
    ServerAuxClass = ServerAux

    MANAGER_CONSTANTS = ManagerConstants(
        server_name='benchmark-server',
        worker_name='benchmark-worker')

    def __init__(self, controller):
        super().__init__(controller)
        self._benchmarker = Benchmarker(self._controller.organizer)
        self._status_dict: dict[int, BenchmarkStatus] = {} # ix -> EvalStatus
        self.is_committee = np.array([])

    def _set_priority(self):
        latest_gen = self._controller.organizer.get_latest_model_generation()
        latest_evaluated_gen = self._latest_evaluated_gen()
        elevate = latest_evaluated_gen + self.benchmark_until_gen_gap < latest_gen
        logger.debug('Benchmark priority: latest_eval_gen=%s, latest_gen=%s, elevate=%s', latest_evaluated_gen, latest_gen, elevate)
        self._controller.set_ratings_priority(elevate)

    def _load_past_data(self):
        self.benchmark_rating_data: BenchmarkRatingData = self._benchmarker.read_ratings_from_db()
        self.is_committee = self.benchmark_rating_data.committee
        logger.debug('Loaded benchmark committee: %s', self.is_committee)

    def _handle_server_disconnect(self, conn: ClientConnection):
        logger.debug('Server disconnected: %s, evaluating ix %s', conn, conn.aux.ix)
        ix = conn.aux.ix
        if ix is not None:
            with self._lock:
                eval_status = self._status_dict[ix].status
                if eval_status != BenchmarkRequestStatus.COMPLETE:
                    self._status_dict[ix].status = BenchmarkRequestStatus.FAILED
                    self._status_dict[ix].owner = None
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            conn.aux.status= ServerStatus.DISCONNECTED
            status_cond.notify_all()

    def _wait_until_work_exists(self):
        with self._lock:
            self._new_work_cond.wait_for(
                lambda: self._latest_evaluated_gen() < self._controller.latest_gen())

    def _latest_evaluated_gen(self) -> Generation:
        latest_gen = 0
        for iagent in self._benchmarker.indexed_agents:
            if (iagent.index < len(self.is_committee)) and self.is_committee[iagent.index]:
                latest_gen = max(latest_gen, iagent.agent.gen)
        return latest_gen

    def _send_match_request(self, conn):
        self._update_status_with_new_matches(conn)
        ix = conn.aux.ix
        if ix is None:
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)
            self._set_priority()
            self._set_ready(conn)
            return

        gen = self._benchmarker.indexed_agents[ix].agent.gen
        status_entry = self._status_dict[ix]
        assert status_entry.status != BenchmarkRequestStatus.COMPLETE
        for opponent_ix in status_entry.ix_match_status:
            match_status = status_entry.ix_match_status[opponent_ix]
            if match_status.status == MatchRequestStatus.PENDING:
                opponent_gen = match_status.opponent_gen
                match_status.status = MatchRequestStatus.REQUESTED
                break

        assert opponent_gen is not None

        data = self._compose_match_request(gen, opponent_gen, ix, opponent_ix)
        conn.socket.send_json(data)

    def _update_status_with_new_matches(self, conn: ClientConnection):
        ix = conn.aux.ix
        ready_for_latest_gen = conn.aux.ready_for_latest_gen
        if ix is not None:
            return

        exclude_agents = self._get_exclude_agents()

        if ready_for_latest_gen:
            latest_gen = self._controller.organizer.get_latest_model_generation()
            latest_agent = self._benchmarker.build_agent(latest_gen, self.n_iters)
            latest_iagent = self._benchmarker._arena._add_agent(
                latest_agent, AgentRole.BENCHMARK, expand_matrix=True,
                db=self._benchmarker._db)

            matches = self._benchmarker.get_unplayed_matches(latest_iagent, self.n_iters,
                                                             exclude_agents=exclude_agents)
            ix = latest_iagent.index
            conn.aux.ready_for_latest_gen = False

        else:
            matches: List[Match] = self._benchmarker.get_next_matches(self.n_iters,
                                                                      self.target_elo_gap,
                                                                      self.n_games,
                                                                      exclude_agents=exclude_agents)
        if not matches:
            self._update_committee()
            conn.aux.ready_for_latest_gen = True
            return

        ix = self._benchmarker.agent_lookup[matches[0].agent1].index

        iagent1 = self._benchmarker.indexed_agents[ix]
        ix_match_status = {}
        for m in matches:
            assert m.agent1 == iagent1.agent
            iagent2 = self._benchmarker.agent_lookup[m.agent2]
            ix_match_status[iagent2.index] = MatchStatus(
                opponent_gen=iagent2.agent.gen,
                n_games=m.n_games,
                status=MatchRequestStatus.PENDING)

        with self._lock:
            self._status_dict[iagent1.index] = BenchmarkStatus(
                mcts_gen=iagent1.agent.gen,
                owner=conn.client_id,
                ix_match_status=ix_match_status,
                status=BenchmarkRequestStatus.REQUESTED)

        conn.aux.ix = ix
        self._set_priority()

    def _compose_match_request(self, gen, opponent_gen, ix, opponent_ix):
        game = self._controller._run_params.game
        tag = self._controller._run_params.tag
        binary = FileToTransfer.from_src_scratch_path(
            source_path=self._controller._organizer.binary_filename,
            scratch_path=f'bin/{game}',
            asset_path_mode='hash')

        files_required = [binary]
        model1 = None
        if gen > 0:
            model1 = FileToTransfer.from_src_scratch_path(
                source_path=self._controller._organizer.get_model_filename(gen),
                scratch_path=f'benchmark-models/{tag}/gen-{gen}.pt',
                asset_path_mode='scratch'
            )
            files_required.append(model1)

        model2 = None
        if opponent_gen > 0:
            model2 = FileToTransfer.from_src_scratch_path(
                source_path=self._controller._organizer.get_model_filename(opponent_gen),
                scratch_path=f'benchmark-models/{tag}/gen-{opponent_gen}.pt',
                asset_path_mode='scratch'
            )
            files_required.append(model2)

        data = {
        'type': 'match-request',
        'agent1': {
            'gen': gen,
            'n_iters': self.n_iters,
            'set_temp_zero': True if gen > 0 else False,
            'tag': tag,
            'binary': binary.scratch_path,
            'model': model1.scratch_path if model1 else None
            },
        'agent2': {
            'gen': opponent_gen,
            'n_iters': self.n_iters,
            'set_temp_zero': True if opponent_gen > 0 else False,
            'tag': tag,
            'binary': binary.scratch_path,
            'model': model2.scratch_path if model2 else None
            },
        'ix1': ix,
        'ix2': opponent_ix,
        'n_games': self.n_games,
        'files_required': [f.to_dict() for f in files_required],
        }

        return data

    def _handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        logger.debug('---Received match result for ix1=%s, ix2=%s, counts=%s', ix1, ix2, counts)

        with self._benchmarker._db.db_lock:
            self._benchmarker._arena.update_match_results(ix1, ix2, counts, MatchType.BENCHMARK, self._benchmarker._db)
        self._benchmarker.refresh_ratings()

        with self._lock:
            self._status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.status == MatchRequestStatus.PENDING for v in self._status_dict[ix1].ix_match_status.values())

        if not has_pending:
            with self._lock:
                self._status_dict[ix1].status = BenchmarkRequestStatus.COMPLETE
                self._status_dict[ix1].owner = None
            conn.aux.ix = None

            exclude_agents = self._get_exclude_agents()
            matches: List[Match] = self._benchmarker.get_next_matches(self.n_iters,
                                                                    self.target_elo_gap,
                                                                    self.n_games,
                                                                    exclude_agents=exclude_agents)
            if not matches:
                self._update_committee()
                conn.aux.ready_for_latest_gen = True

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.release_lock(conn.client_domain)
        self._set_priority()

    def _update_committee(self):
        self._benchmarker.refresh_ratings()
        committee = self._benchmarker.select_committee(self.target_elo_gap)
        self.is_committee = committee
        with self._benchmarker._db.db_lock:
            self._benchmarker._db.commit_ratings(self._benchmarker.indexed_agents,
                                                    self._benchmarker._arena.ratings,
                                                    committee=committee)

    def _get_exclude_agents(self):
        if len(self.is_committee) == 0:
            return np.array([], dtype=int)

        exclude_agents = np.where(self.is_committee == False)[0]
        exclude_agents = np.concatenate([exclude_agents,
                                         np.arange(len(self.is_committee),
                                                   len(self._benchmarker.indexed_agents) - len(self.is_committee))])
        return exclude_agents

    @property
    def n_games(self):
        return self._controller.params.n_games_per_benchmark

    @property
    def n_iters(self):
        return self._controller.params.agent_n_iters

    @property
    def target_elo_gap(self):
        return self._controller.params.target_elo_gap

    @property
    def benchmark_until_gen_gap(self):
        return self._controller.params.benchmark_until_gen_gap