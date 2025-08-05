from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import AgentRole, IndexedAgent, Match, MatchType
from alphazero.logic.self_evaluator import BenchmarkRatingData, SelfEvaluator
from alphazero.logic.custom_types import ClientConnection, ClientId, Domain, FileToTransfer, \
    Generation, ServerStatus
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.servers.loop_control.gaming_manager_base import GamingManagerBase, ManagerConfig, \
    ServerAuxBase, WorkerAux
from util.index_set import IndexSet
from util.socket_util import JsonDict


from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


class MatchRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    PENDING = 'pending'


class SelfEvalRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    FAILED = 'failed'


@dataclass
class MatchStatus:
    opponent_gen: Generation
    n_games: int
    status: MatchRequestStatus

    def pending(self) -> bool:
        return self.status == MatchRequestStatus.PENDING


@dataclass
class SelfEvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: dict[int, MatchStatus] = field(default_factory=dict)  # ix -> MatchStatus
    status: Optional[SelfEvalRequestStatus] = None


@dataclass
class SelfEvalServerAux(ServerAuxBase):
    """
    Auxiliary data stored per server connection.
    """
    ix: Optional[int] = None  # test agent index
    ready_for_latest_gen: bool = False

    def work_in_progress(self) -> bool:
        return self.ix is not None


class SelfEvalManager(GamingManagerBase):
    """
    SelfEvalManager is responsible for managing the lifecycle of the self-evaluating process.

    ### Self-evaluation Cadence & Priority

    Self-evaluation is usually scheduled at a fixed cadence (e.g., every 25 generations), and runs
    when the rating domain (used by self-eval) is elevated above other domains. However, it can
    also run earlier if other domains lower their priority below the rating domain’s default.

    ### What a Benchmark is

    A benchmark is a curated committee of agents whose Elo ratings differ by at least a predefined
    threshold. This ensures the committee spans a representative spectrum of skill levels. Committee
    selection is handled through the self-eval process, using match results and Elo gap analysis.

    ### Progressive Self-evaluation

    Self-evaluation is performed progressively. For example:

    - When self-evaluating generations 0 to 26, the system may select a set of 8 agents to form a
        committee.
    - At a later generation (e.g., gen-48), it self-evaluates gen-48 against the committee.
    - It then introduces new agents that help close the largest Elo gaps. These agents play matches
        against:
        - The committee
        - Gen-48 and any agents previously introduced (e.g., gen-37 if added earlier)

    This process continues until all Elo gaps fall below the target threshold or are indivisible
    (e.g., gen-0 and gen-1 might still have a large gap that cannot be split further).

    ### Robustness & Continuity

    SelfEvalManager can recover from interruptions. If a benchmark is incomplete, it can resume
    where it left off, skipping redundant matches and continuing until the benchmark is complete.

    """

    def __init__(self, controller: LoopController):
        manager_config = ManagerConfig(
            worker_aux_class=WorkerAux,
            server_aux_class=SelfEvalServerAux,
            server_name='self-eval-server',
            worker_name='self-eval-worker',
            domain=Domain.SELF_EVAL,
            )
        super().__init__(controller, manager_config)
        self._self_evaluator = SelfEvaluator(self._controller.organizer)
        self._status_dict: dict[int, SelfEvalStatus] = {}  # ix -> EvalStatus
        self.excluded_agent_indices: IndexSet = IndexSet()
        self.evaluated_iagents: List[IndexedAgent] = []

    def set_priority(self):
        latest_gen = self._controller.organizer.get_latest_model_generation()
        latest_evaluated_gen = self.num_evaluated_gens()
        if latest_gen is None:
            elevate = False
        else:
            elevate = latest_evaluated_gen + self.self_eval_until_gen_gap < latest_gen
        logger.debug('Self-evaluation priority: latest_eval_gen=%s, latest_gen=%s, elevate=%s',
                     latest_evaluated_gen, latest_gen, elevate)
        self._controller.set_domain_priority(self._config.domain, elevate)

    def load_past_data(self):
        benchmark_rating_data: BenchmarkRatingData = self._self_evaluator.read_ratings_from_db()
        self.evaluated_iagents = benchmark_rating_data.iagents
        self.excluded_agent_indices = ~benchmark_rating_data.committee

    def num_evaluated_gens(self):
        gen = 0
        for iagent in self.evaluated_iagents:
            if iagent.agent.gen > gen:
                gen = iagent.agent.gen
        return gen

    def handle_server_disconnect(self, conn: ClientConnection):
        logger.debug('Server disconnected: %s, evaluating ix %s', conn, conn.aux.ix)
        ix = conn.aux.ix
        if ix is not None:
            with self._lock:
                eval_status = self._status_dict[ix].status
                if eval_status != SelfEvalRequestStatus.COMPLETE:
                    self._status_dict[ix].status = SelfEvalRequestStatus.FAILED
                    self._status_dict[ix].owner = None
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            conn.aux.status = ServerStatus.DISCONNECTED
            status_cond.notify_all()

    def send_match_request(self, conn):
        self._self_evaluator.refresh_ratings()
        self._update_status_with_new_matches(conn)
        ix = conn.aux.ix
        if ix is None:
            self._controller.set_domain_priority(self._config.domain, False)
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)
            self._set_ready(conn)
            logger.info('DONE. No matches available for %s', conn)
            return

        gen = self._self_evaluator.indexed_agents[ix].agent.gen
        status_entry = self._status_dict[ix]
        assert status_entry.status != SelfEvalRequestStatus.COMPLETE
        for opponent_ix in status_entry.ix_match_status:
            match_status = status_entry.ix_match_status[opponent_ix]
            if match_status.status == MatchRequestStatus.PENDING:
                opponent_gen = match_status.opponent_gen
                match_status.status = MatchRequestStatus.REQUESTED
                break

        assert opponent_gen is not None

        data = self._compose_match_request(gen, opponent_gen, ix, opponent_ix)
        conn.socket.send_json(data)

    def handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        logger.debug('---Received match result for ix1=%s, ix2=%s, counts=%s', ix1, ix2, counts)

        with self._self_evaluator.db.db_lock:
            self._self_evaluator._arena.update_match_results(ix1, ix2, counts, MatchType.BENCHMARK,
                                                          self._self_evaluator.db)
        self._self_evaluator.refresh_ratings()

        with self._lock:
            status = self._status_dict[ix1]
            status.ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.pending() for v in status.ix_match_status.values())

        if not has_pending:
            self._update_committee()
            with self._lock:
                self._status_dict[ix1].status = SelfEvalRequestStatus.COMPLETE
                self._status_dict[ix1].owner = None
            conn.aux.ix = None

            matches: List[Match] = self._self_evaluator.get_next_matches(
                    self.n_iters, self.target_elo_gap, self.n_games,
                    excluded_indices=self.excluded_agent_indices)
            if not matches:
                conn.aux.ready_for_latest_gen = True

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.release_lock(conn.client_domain)
        self.set_priority()

    def _update_status_with_new_matches(self, conn: ClientConnection):
        ix = conn.aux.ix
        ready_for_latest_gen = conn.aux.ready_for_latest_gen
        if ix is not None:
            return
        if ready_for_latest_gen:
            latest_gen = self._controller.organizer.get_latest_model_generation()
            latest_agent = self._self_evaluator.build_agent(latest_gen, self.n_iters)
            latest_iagent = self._self_evaluator._arena.add_agent(
                latest_agent, {AgentRole.BENCHMARK}, expand_matrix=True, db=self._self_evaluator.db)

            matches = self._self_evaluator.get_unplayed_matches(
                    latest_iagent, self.n_iters, excluded_indices=self.excluded_agent_indices)
            ix = latest_iagent.index
            conn.aux.ready_for_latest_gen = False

        else:
            matches: List[Match] = self._self_evaluator.get_next_matches(
                    self.n_iters, self.target_elo_gap, self.n_games,
                    excluded_indices=self.excluded_agent_indices)

        if not matches:
            self._update_committee()
            conn.aux.ready_for_latest_gen = True
            return

        ix = self._self_evaluator.agent_lookup[matches[0].agent1].index

        iagent1 = self._self_evaluator.indexed_agents[ix]
        ix_match_status = {}
        for m in matches:
            assert m.agent1 == iagent1.agent
            iagent2 = self._self_evaluator.agent_lookup[m.agent2]
            ix_match_status[iagent2.index] = MatchStatus(
                opponent_gen=iagent2.agent.gen,
                n_games=m.n_games,
                status=MatchRequestStatus.PENDING)

        with self._lock:
            self._status_dict[iagent1.index] = SelfEvalStatus(
                mcts_gen=iagent1.agent.gen,
                owner=conn.client_id,
                ix_match_status=ix_match_status,
                status=SelfEvalRequestStatus.REQUESTED)

        conn.aux.ix = ix
        self.set_priority()

    def _compose_match_request(self, gen, opponent_gen, ix, opponent_ix):
        game = self._controller._run_params.game
        tag = self._controller._run_params.tag

        binary_path = self._controller._get_binary_path()
        binary = FileToTransfer.from_src_scratch_path(
            source_path=binary_path,
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
                'type': 'MCTS',
                'data': {
                    'gen': gen,
                    'n_iters': self.n_iters,
                    'set_temp_zero': True if gen > 0 else False,
                    'tag': tag,
                    'binary': binary.scratch_path,
                    'model': model1.scratch_path if model1 else None
                    }
                },
            'agent2': {
                'type': 'MCTS',
                'data': {
                    'gen': opponent_gen,
                    'n_iters': self.n_iters,
                    'set_temp_zero': True if opponent_gen > 0 else False,
                    'tag': tag,
                    'binary': binary.scratch_path,
                    'model': model2.scratch_path if model2 else None
                    }
                },
            'ix1': ix,
            'ix2': opponent_ix,
            'n_games': self.n_games,
            'files_required': [f.to_dict() for f in files_required],
            }
        logger.info(f"Self-evaluating request ix: {data['ix1']} vs {data['ix2']}, "
                    f"gen: {data['agent1']['data']['gen']} vs {data['agent2']['data']['gen']}, "
                    f"n_games: {data['n_games']}")
        return data

    def _latest_evaluated_gen(self) -> Generation:
        latest_gen = 0
        for iagent in self._self_evaluator.indexed_agents:
            excluded_agent_not_empty: bool = len(self.excluded_agent_indices) > 0
            if excluded_agent_not_empty and iagent.index in ~self.excluded_agent_indices:
                latest_gen = max(latest_gen, iagent.agent.gen)
        return latest_gen

    def _update_committee(self):
        if self._self_evaluator.has_no_matches():
            return
        self._self_evaluator.refresh_ratings()
        committee: IndexSet = SelfEvaluator.select_committee(
                self._self_evaluator.ratings, self.target_elo_gap)
        self.excluded_agent_indices = ~committee
        self.evaluated_iagents = self._self_evaluator.indexed_agents
        with self._self_evaluator.db.db_lock:
            self._self_evaluator.db.commit_ratings(
                    self._self_evaluator.indexed_agents, self._self_evaluator._arena.ratings,
                    committee=committee)
        committee_gens = [self._self_evaluator.indexed_agents[i].agent.gen for i in committee]
        logger.info(f"Benchmark committee: {committee_gens}")

    def _task_finished(self) -> bool:
        latest_gen = self._controller._organizer.get_latest_model_generation(default=0)
        has_new_gen = self.num_evaluated_gens() < latest_gen
        logger.debug(f"has_new_gen: {has_new_gen}, {self.num_evaluated_gens()}, latest: {latest_gen}")
        if has_new_gen:
            return False
        else:
            matches: List[Match] = self._self_evaluator.get_next_matches(
                    self.n_iters, self.target_elo_gap, self.n_games,
                    excluded_indices=self.excluded_agent_indices)
            logger.debug(f"matches: {matches}")
            if len(matches) > 0:
                return False

        logger.info("Self-evaluation is complete.")
        return True

    @property
    def n_games(self):
        return self._controller.rating_params.n_games_per_self_evaluation

    @property
    def n_iters(self):
        return self._controller.rating_params.rating_player_options.num_iterations

    @property
    def target_elo_gap(self):
        if self._controller.rating_params.target_elo_gap is None:
            return self._controller.rating_params.default_target_elo_gap.first_run
        return self._controller.rating_params.target_elo_gap

    @property
    def self_eval_until_gen_gap(self):
        return self._controller.params.self_eval_until_gen_gap
