from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import MCTSAgent, AgentRole
from alphazero.logic.custom_types import ClientConnection, Generation, EvalTag, ServerStatus, ClientId
from alphazero.logic.evaluator import Evaluator, EvalUtils
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.match_runner import MatchType
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = get_logger()


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
    opponent_gen: Generation
    n_games: int
    status: MatchRequestStatus


@dataclass
class EvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: Dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    status: Optional[EvalRequestStatus] = None


class EvalManager:
    """
    A separate EvalManager is created for each rating-tag.
    """

    @dataclass
    class ServerAux:
        """
        Auxiliary data stored per server connection.
        """
        status_cond: threading.Condition = field(default_factory=threading.Condition)
        status: ServerStatus = ServerStatus.BLOCKED
        needs_new_opponents: bool = True
        estimated_rating: Optional[float] = None
        ix: Optional[int] = None  # test agent index

    @dataclass
    class WorkerAux:
        """
        Auxiliary data stored per worker connection.
        """
        cond: threading.Condition = field(default_factory=threading.Condition)
        pending_pause_ack: bool = False
        pending_unpause_ack: bool = False

    def __init__(self, controller: LoopController, tag: EvalTag):
        self._tag = tag
        self._controller = controller

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cond = threading.Condition(self._lock)
        self._evaluator = Evaluator(self._controller._organizer)
        self._eval_status_dict: Dict[int, EvalStatus] = {} # ix -> EvalStatus

    def add_server(self, conn: ClientConnection):
        conn.aux = EvalManager.ServerAux()
        self._controller.send_handshake_ack(conn)
        assets_request = conn.socket.recv_json()
        assert assets_request['type'] == 'assets-request'

        for asset in assets_request['assets']:
            conn.socket.send_file(asset)

        self._start()
        logger.info('Starting eval-recv-loop for %s...', conn)
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'eval-server',
            disconnect_handler=self._handle_server_disconnect)

        thread = threading.Thread(target=self._manage_server, args=(conn,),
                                  daemon=True, name=f'manage-eval-server')
        thread.start()

    def notify_of_new_model(self):
        """
        Notify manager that there is new work to do.
        """
        self._set_priority()
        with self._lock:
            self._new_work_cond.notify_all()

    def _set_priority(self):
        latest_gen = self._controller.latest_gen()
        dict_len = len(self._eval_status_dict)
        eval_in_progress = any(data.status == EvalRequestStatus.REQUESTED for data in self._eval_status_dict.values())

        target_rate = self._controller.params.target_rating_rate
        num = dict_len + (0 if eval_in_progress else 1)
        den = max(1, latest_gen)
        current_rate = num / den

        elevate = current_rate < target_rate
        logger.debug('Ratings elevate-priority:%s (latest=%s, dict_len=%s, in_progress=%s, '
                     'current=%.2f, target=%.2f)', elevate, latest_gen, dict_len,
                     eval_in_progress, current_rate, target_rate)
        #TODO: separate out eval from rating domain
        self._controller.set_ratings_priority(elevate)
        for table in self._controller._gpu_contention_manager._get_all_tables():
            logger.info(f'Table priority: {table}')

    def _start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._load_past_data()

    def _load_past_data(self):
        logger.info('Loading past ratings data...')
        rating_data = self._evaluator.read_ratings_from_db()
        evaluated_ixs = [iagent.index for iagent in rating_data.evaluated_iagents]
        for test_ix in self._evaluator.test_agent_ixs():
            gen = self._evaluator.indexed_agents[test_ix].agent.gen
            ix_match_status = self._load_ix_match_status(test_ix)
            self._eval_status_dict[test_ix] = EvalStatus(mcts_gen=gen, ix_match_status=ix_match_status)
            if test_ix not in evaluated_ixs:
                self._eval_status_dict[test_ix].status = EvalRequestStatus.FAILED
            else:
                self._eval_status_dict[test_ix].status = EvalRequestStatus.COMPLETE

        self._set_priority()

    def _handle_server_disconnect(self, conn: ClientConnection):
        logger.info('Server disconnected: %s, evaluating ix %s', conn, conn.aux.ix)
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

    def _wait_for_unblock(self, conn: ClientConnection) -> ServerStatus:
        """
        The server status is initially BLOCKED. This function waits until that status is
        changed (either to READY or DISCONNECTED). After waiting, it resets the status to
        BLOCKED, and returns what the status was changed to.
        """
        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            status_cond.wait_for(lambda: conn.aux.status != ServerStatus.BLOCKED)
            status = conn.aux.status
            conn.aux.status = ServerStatus.BLOCKED
            return status

    def _wait_until_work_exists(self):
        with self._lock:
            self._new_work_cond.wait_for(
                lambda: len(self._eval_status_dict) < self._controller.latest_gen())

    def _send_match_request(self, conn: ClientConnection):
        #TODO: remove this assert and implement mechasnim to request for binary, extra dependencies or model files
        assert conn.is_on_localhost()
        ix = conn.aux.ix
        if ix is None:
            gen = self._get_next_gen_to_eval()
            assert gen is not None
            test_agent = MCTSAgent(gen, n_iters=self.n_iters, set_temp_zero=True, tag=self._controller._organizer.tag)
            test_iagent = self._evaluator.add_agent(test_agent, AgentRole.TEST, expand_matrix=True, db=self._evaluator._db)
            conn.aux.ix = test_iagent.index
            with self._lock:
                if test_iagent.index in self._eval_status_dict:
                    assert self._eval_status_dict[test_iagent.index].status == EvalRequestStatus.FAILED
                    self._eval_status_dict[test_iagent.index].status = EvalRequestStatus.REQUESTED
                else:
                    self._eval_status_dict[test_iagent.index] = EvalStatus(mcts_gen=gen, owner=conn.client_id,
                                                                           status=EvalRequestStatus.REQUESTED)
            conn.aux.needs_new_opponents = True
            self._set_priority()
        else:
            test_iagent = self._evaluator.indexed_agents[ix]

        logger.info('***Evaluating gen %s', test_iagent.agent.gen)
        estimated_rating = conn.aux.estimated_rating
        if estimated_rating is None:
            estimated_rating = self._estimate_rating(test_iagent)
            logger.info('Estimated rating for gen %s: %s', test_iagent.agent.gen, estimated_rating)
            conn.aux.estimated_rating = estimated_rating

        n_games_in_progress = sum([data.n_games for data in self._eval_status_dict[test_iagent.index].ix_match_status.values() \
            if data.status in (MatchRequestStatus.COMPLETE, MatchRequestStatus.REQUESTED)])
        n_games_needed = self.n_games - n_games_in_progress
        assert n_games_needed > 0, f"{self.n_games} games needed, but {n_games_in_progress} already in progress"

        need_new_opponents = conn.aux.needs_new_opponents
        if need_new_opponents:
            logger.info('Requesting %s games for gen %s, estimated rating: %s', n_games_needed, test_iagent.agent.gen, estimated_rating)
            opponent_ixs_played = [ix for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() \
                if data.status in (MatchRequestStatus.COMPLETE, MatchRequestStatus.REQUESTED)]
            chosen_ixs, num_matches = self._evaluator.gen_matches(estimated_rating, opponent_ixs_played, n_games_needed)
            with self._lock:
                self._update_eval_status(test_iagent.index, chosen_ixs, num_matches)
            conn.aux.needs_new_opponents = False

        candidates = [(ix, data.n_games) for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() if data.status == MatchRequestStatus.PENDING]
        next_opponent_ix, next_n_games = sorted(candidates, key=lambda x: x[1])[-1]
        next_opponent_agent = self._evaluator.indexed_agents[next_opponent_ix].agent
        data = {
            'type': 'match-request',
            'agent1': {
                'gen': test_iagent.agent.gen,
                'n_iters': self.n_iters,
                'set_temp_zero': True,
                'tag': self._controller._organizer.tag,
            },
            'agent2': {
                'gen': next_opponent_agent.gen,
                'n_iters': next_opponent_agent.n_iters,
                'set_temp_zero': next_opponent_agent.set_temp_zero,
                'tag': next_opponent_agent.tag,

            },
            'ix1': test_iagent.index,
            'ix2': int(next_opponent_ix),
            'n_games': int(next_n_games),
        }
        conn.socket.send_json(data)
        self._eval_status_dict[test_iagent.index].ix_match_status[next_opponent_ix].status = MatchRequestStatus.REQUESTED

    def _get_next_gen_to_eval(self):
        latest_gen = self._controller.latest_gen()
        evaluated_gens = [data.mcts_gen for data in self._eval_status_dict.values() \
            if data.status in (EvalRequestStatus.COMPLETE, EvalRequestStatus.REQUESTED)]
        target_rating_rate = self._controller.params.target_rating_rate
        gen = EvalUtils.get_next_gen_to_eval(latest_gen, evaluated_gens, target_rating_rate)
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
                logger.info('...set match between %s and %s to obsolete', self._evaluator.indexed_agents[test_ix].index, match_status.opponent_gen)

        for ix, n_games in zip(chosen_ixs, num_matches):
            if ix not in self._eval_status_dict[test_ix].ix_match_status:
                new_opponent_gen = self._evaluator.indexed_agents[ix].agent.gen
                self._eval_status_dict[test_ix].ix_match_status[ix] = \
                    MatchStatus(new_opponent_gen, n_games, MatchRequestStatus.PENDING)
                logger.info('+++add new %s matches between %s and %s', n_games, self._evaluator.indexed_agents[test_ix].index, ix)
            else:
                self._eval_status_dict[test_ix].ix_match_status[ix].n_games = n_games
                if self._eval_status_dict[test_ix].ix_match_status[ix].status == MatchRequestStatus.OBSOLETE:
                    self._eval_status_dict[test_ix].ix_match_status[ix].status = MatchRequestStatus.PENDING
                logger.info('+++modify to %s matches between %s and %s', n_games, self._evaluator.indexed_agents[test_ix].index, ix)

    def _load_ix_match_status(self, test_ix: int) -> Dict[int, MatchStatus]:
        W_matrix = self._evaluator._arena._W_matrix
        n_games_played = W_matrix[test_ix, :] + W_matrix[:, test_ix]
        match_status_dict = {}
        for ix, n_games in enumerate(n_games_played):
            if ix == test_ix:
                continue
            if n_games > 0:
                gen = self._evaluator.indexed_agents[ix].agent.gen
                match_status_dict[ix] = MatchStatus(gen, int(n_games), MatchRequestStatus.COMPLETE)
        return match_status_dict

    def _manage_server(self, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id
            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            table.activate(domain)

            # NOTE: the worker loop breaks when the table becomes DEACTIVATING, while this loop
            # only breaks when the table becomes INACTIVE. It is important then to use
            # (not inactive) in the below loop-condition, rather than (active).
            while not table.inactive(domain):
                status = self._wait_for_unblock(conn)
                if status == ServerStatus.DISCONNECTED:
                    break
                if conn.aux.ix is None:
                    self._wait_until_work_exists()

                logger.info(f"Managing eval-server, priority: {table}")
                table.activate(domain)
                if not table.acquire_lock(domain):
                    break
                self._send_match_request(conn)

                # We do not release the lock here. The lock is released either when a gen is
                # fully rated, or when the server disconnects.
        except SocketSendException:
            logger.warning('Error sending to %s - server likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _server_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('eval-server received json message: %s', msg)

        if msg_type == 'ready':
            self._handle_ready(conn)
        elif msg_type == 'log-sync-start':
            self._controller.start_log_sync(conn, msg['log_filename'])
        elif msg_type == 'log-sync-stop':
            self._controller.stop_log_sync(conn, msg['log_filename'])
        elif msg_type == 'match-result':
            self._handle_match_result(msg, conn)
        else:
            logger.warning('eval-server: unknown message type: %s', msg)
        return False

    def _handle_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            conn.aux.status = ServerStatus.READY
            status_cond.notify_all()

    def _handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        logger.info('---Received match result for ix1=%s, ix2=%s, counts=%s', ix1, ix2, counts)
        with self._evaluator._db.db_lock:
            self._evaluator._arena.update_match_results(ix1, ix2, counts, MatchType.EVALUATE, self._evaluator._db)
        self._evaluator.refresh_ratings()
        new_rating = self._evaluator.arena_ratings[ix1]
        old_rating = conn.aux.estimated_rating
        logger.info('Old rating: %s, New rating: %s', old_rating, new_rating)
        if abs(new_rating - old_rating) > self.error_threshold:
            logger.info('Rating change too large, requesting new opponents...')
            conn.aux.needs_new_opponents = True
            conn.aux.estimated_rating = new_rating

        with self._lock:
            self._eval_status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.status == MatchRequestStatus.PENDING for v in self._eval_status_dict[ix1].ix_match_status.values())

        if not has_pending:
            assert self._evaluator._arena.n_games_played(self._evaluator.indexed_agents[ix1].agent) == self.n_games
            self._eval_status_dict[ix1].status = EvalRequestStatus.COMPLETE
            self._eval_status_dict[ix1].owner = None
            ix = conn.aux.ix
            assert ix == ix1
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)

            test_ixs, interpolated_ratings = self._evaluator.interpolate_ratings()
            test_iagents = [self._evaluator.indexed_agents[ix] for ix in test_ixs]
            with self._evaluator._db.db_lock:
                self._evaluator._db.commit_ratings(test_iagents, interpolated_ratings)
            conn.aux.estimated_rating = None
            conn.aux.ix = None
            logger.info('///Finished evaluating gen %s, rating: %s',
                        self._evaluator.indexed_agents[ix1].agent.gen,
                        interpolated_ratings[np.where(test_ixs == ix1)[0]])

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.release_lock(conn.client_domain)
        self._set_priority()

    def add_worker(self, conn: ClientConnection):
        conn.aux = EvalManager.WorkerAux()

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'eval-worker',
            disconnect_handler=self._handle_worker_disconnect)

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('eval-worker received json message: %s', msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'worker-ready':
            self._handle_worker_ready(conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('eval-worker: unknown message type: %s', msg)
        return False

    def _handle_worker_ready(self, conn: ClientConnection):
        logger.debug( 'Eval-Worker %s is ready', conn)
        thread = threading.Thread(target=self._manage_worker, args=(conn,),
                                  daemon=True, name=f'manage-ratings-worker')
        thread.start()

    def _manage_worker(self, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id

            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            self._pause(conn)

            while table.active(domain):
                if not table.acquire_lock(domain):
                    break
                self._unpause(conn)
                if table.wait_for_lock_expiry(domain):
                    self._pause(conn)
                    table.release_lock(domain)
        except SocketSendException:
            logger.warning('Error sending to %s - worker likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _handle_worker_disconnect(self, conn: ClientConnection):
        aux: EvalManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

        # We set the management status to DEACTIVATING, rather than INACTIVE, here, so that the
        # worker loop breaks while the server loop continues.
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.mark_as_deactivating(conn.client_domain)

    def _pause(self, conn: ClientConnection):
        logger.debug('Pausing %s...', conn)

        aux: EvalManager.WorkerAux = conn.aux
        aux.pending_pause_ack = True

        conn.socket.send_json({ 'type': 'pause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_pause_ack)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)

        aux: EvalManager.WorkerAux = conn.aux
        aux.pending_unpause_ack = True

        conn.socket.send_json({ 'type': 'unpause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_unpause_ack)

        logger.debug('Unpause of %s complete!', conn)

    def _handle_pause_ack(self, conn: ClientConnection):
        aux: EvalManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        aux: EvalManager.WorkerAux = conn.aux
        with aux.cond:
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

    def _update_weights(self, gen: Generation, conn: ClientConnection):
        self._controller.broadcast_weights(conn, gen)

    @property
    def n_games(self):
        return self._controller.params.n_games_per_evaluation

    @property
    def n_iters(self):
        return self._controller.params.eval_agent_n_iters

    @property
    def error_threshold(self):
        return self._controller.params.eval_error_threshold
