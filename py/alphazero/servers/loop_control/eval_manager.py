from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import MCTSAgent, AgentRole
from alphazero.logic.arena import Arena
from alphazero.logic.custom_types import ClientConnection, Generation, EvalTag, ServerStatus, ClientId
from alphazero.logic.evaluator import Evaluator, EvalUtils
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.match_runner import MatchType
from util.logging_util import get_logger
from util.py_util import find_largest_gap
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


@dataclass
class MatchStatus:
    opponent_gen: Generation
    n_games: int
    status: MatchRequestStatus
    estimated_rating: Optional[float] = None


@dataclass
class EvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: Dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    is_done: bool = False


class EvalManager:
    """
    A separate EvalManager is created for each rating-tag.
    """
    def __init__(self, controller: LoopController, tag: EvalTag):
        self._tag = tag
        self._controller = controller

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cond = threading.Condition(self._lock)
        self._evaluator = Evaluator(self._controller._organizer)
        self._eval_status_dict: Dict[int, EvalStatus] = {} # ix -> EvalStatus

        self.n_games = 1000
        self.error_threshold = 100
        self.n_iters = 100

    def add_server(self, conn: ClientConnection):
        ssh_pub_key = ssh_util.get_pub_key()
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
            'tag': self._controller.run_params.tag,
            'ssh_pub_key': ssh_pub_key,
            'on_ephemeral_local_disk_env': self._controller.on_ephemeral_local_disk_env,
            'asset-requirements': self._controller.get_asset_requirements(),
        }
        conn.socket.send_json(reply)

        assets_request = conn.socket.recv_json()
        assert assets_request['type'] == 'assets-request'
        for asset in assets_request['assets']:
            conn.socket.send_file(asset)

        conn.aux['status_cond'] = threading.Condition()
        conn.aux['status'] = ServerStatus.BLOCKED
        conn.aux['needs_new_opponents_cond'] = threading.Condition()
        conn.aux['needs_new_opponents'] = True

        self._start()
        logger.info('Starting eval-recv-loop for %s...', conn)
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'eval-server',
            disconnect_handler=self._handle_server_disconnect)

        thread = threading.Thread(target=self._manage_server, args=(conn,),
                                  daemon=True, name=f'manage-eval-server')
        thread.start()

    def add_worker(self, conn: ClientConnection):
        conn.aux['ack_cond'] = threading.Condition()

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'eval-worker',
            disconnect_handler=self._handle_worker_disconnect)

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
        eval_in_progress = any(not data.is_done for data in self._eval_status_dict.values())

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

    def _start(self):
        with self._lock:
            if self._started:
                logger.info('****!!!!!!!!**** EvalManager already started ****!!!!!!!!****')
                return
            self._started = True
            self._load_past_data()

    def _load_past_data(self):
        logger.info('Loading past ratings data...')
        rating_data = self._evaluator.read_ratings_from_db()
        self._eval_status_dict = {self._evaluator.agent_lookup[agent].index: \
            EvalStatus(mcts_gen=agent.gen, is_done=True) \
                for agent in rating_data.evaluated_agents}
        self._set_priority()

    def _handle_server_disconnect(self, conn: ClientConnection):
        gen = conn.aux.pop('gen', None)
        if gen is not None:
            with self._lock:
                eval_status = [data for data in self._eval_status_dict.values() if data.mcts_gen == gen]
                if not eval_status:
                    eval_status[0].owner = None

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.DISCONNECTED
            status_cond.notify_all()

    def _handle_worker_disconnect(self, conn: ClientConnection):
        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

        # We set the management status to DEACTIVATING, rather than INACTIVE, here, so that the
        # worker loop breaks while the server loop continues.
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.mark_as_deactivating(conn.client_domain)

    def _wait_for_unblock(self, conn: ClientConnection) -> ServerStatus:
        """
        The server status is initially BLOCKED. This function waits until that status is
        changed (either to READY or DISCONNECTED). After waiting, it resets the status to
        BLOCKED, and returns what the status was changed to.
        """
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            status_cond.wait_for(lambda: conn.aux['status'] != ServerStatus.BLOCKED)
            status = conn.aux['status']
            conn.aux['status'] = ServerStatus.BLOCKED
            return status

    def _wait_until_work_exists(self):
        with self._lock:
            self._new_work_cond.wait_for(
                lambda: len(self._eval_status_dict) < self._controller.latest_gen())

    def _send_match_request(self, conn: ClientConnection):
        assert conn.is_on_localhost()
        gen = conn.aux.get('gen', None)
        evaluated_gens = [data.mcts_gen for data in self._eval_status_dict.values()]
        ratings = self._evaluator.arena_ratings[list(self._eval_status_dict.keys())]

        if gen is None:
            latest_gen = self._controller.latest_gen()
            target_rating_rate = self._controller.params.target_rating_rate
            with self._lock:
                gen = EvalUtils.get_next_gen_to_eval(latest_gen, evaluated_gens, target_rating_rate)
                assert gen is not None
                conn.aux['gen'] = gen

                test_agent = MCTSAgent(gen, n_iters=self.n_iters, set_temp_zero=True,
                                       tag=self._controller._organizer.tag)
                test_iagent = self._evaluator._arena._add_agent(test_agent,
                                                                AgentRole.TEST,
                                                                expand_matrix=True,
                                                                db=self._evaluator._db)
                self._eval_status_dict[test_iagent.index] = EvalStatus(mcts_gen=gen,
                                                                       owner=conn.client_id)

            cond = conn.aux['needs_new_opponents_cond']
            with cond:
                conn.aux['needs_new_opponents'] = True

            self._set_priority()
        else:
            test_iagent = [iagent for iagent in self._evaluator.indexed_agents if \
                iagent.role == AgentRole.TEST and iagent.agent.gen == gen][0]


        estimated_rating = EvalUtils.estimate_rating_nearby_gens(gen, evaluated_gens, ratings)
        if estimated_rating is None:
            estimated_rating = np.mean(self._evaluator.arena_ratings)

        n_games_played = self._evaluator._arena.n_games_played(test_iagent.agent)
        n_games_needed = self.n_games - n_games_played

        if conn.aux['needs_new_opponents'] and n_games_needed > 0:
            chosen_ixs, num_matches = self._evaluator.gen_matches(test_iagent.agent, estimated_rating, n_games_needed)
            for ix, match_status in self._eval_status_dict[test_iagent.index].ix_match_status.items():
                if ix not in chosen_ixs and match_status.status == MatchRequestStatus.PENDING:
                    match_status.status = MatchRequestStatus.OBSOLETE
                    logger.info('set match between %s and %s to obsolete', gen, match_status.opponent_gen)
                elif ix in chosen_ixs and match_status.status == MatchRequestStatus.OBSOLETE:
                    match_status.status = MatchRequestStatus.PENDING
                    logger.info('set match between %s and %s to pending', gen, match_status.opponent_gen)

            for ix, n_games in zip(chosen_ixs, num_matches):
                if ix not in self._eval_status_dict[test_iagent.index].ix_match_status:
                    new_opponent_gen = self._evaluator.indexed_agents[ix].agent.gen
                    self._eval_status_dict[test_iagent.index].ix_match_status[ix] = \
                        MatchStatus(new_opponent_gen, n_games, MatchRequestStatus.PENDING, estimated_rating)
                    logger.info('add new pending match between %s and %s', gen, new_opponent_gen)
            cond = conn.aux['needs_new_opponents_cond']
            with cond:
                conn.aux['needs_new_opponents'] = False

        candidates = [(ix, data.n_games) for ix, data in self._eval_status_dict[test_iagent.index].ix_match_status.items() if data.status == MatchRequestStatus.PENDING]
        next_opponent_ix, next_n_games = sorted(candidates, key=lambda x: x[1])[-1]
        next_opponent_agent = self._evaluator.indexed_agents[next_opponent_ix].agent
        data = {
            'type': 'match-request',
            'agent1': {
                'ix': test_iagent.index,
                'gen': test_iagent.agent.gen,
                'n_iters': self.n_iters,
                'set_temp_zero': True,
                'tag': self._controller._organizer.tag,
            },
            'agent2': {
                'ix': int(next_opponent_ix),
                'gen': next_opponent_agent.gen,
                'n_iters': next_opponent_agent.n_iters,
                'set_temp_zero': next_opponent_agent.set_temp_zero,
                'tag': next_opponent_agent.tag,
            },
            'n_games': int(next_n_games),
        }
        conn.socket.send_json(data)

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
                if conn.aux.get('gen', None) is None:
                    self._wait_until_work_exists()

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

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('eval-worker received json message: %s', msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('eval-worker: unknown message type: %s', msg)
        return False

    def _handle_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.READY
            status_cond.notify_all()

    def _handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        logger.info('Received match result %s from %s', msg, conn)
        ix1 = msg['ix1']
        ix2 = msg['ix2']
        counts = WinLossDrawCounts.from_json(msg['record'])
        with self._evaluator._db.db_lock:
            self._evaluator._arena.update_match_results(ix1, ix2, counts, MatchType.EVALUATE, self._evaluator._db)
        self._evaluator.refresh_ratings()
        new_rating = self._evaluator.arena_ratings[ix1]
        old_rating = self._eval_status_dict[ix1].ix_match_status[ix2].estimated_rating
        if abs(new_rating - old_rating) > self.error_threshold:
            cond = conn.aux['needs_new_opponents_cond']
            with cond:
                conn.aux['needs_new_opponents'] = True

        with self._lock:
            self._eval_status_dict[ix1].ix_match_status[ix2].status = MatchRequestStatus.COMPLETE
            has_pending = any(v.status == MatchRequestStatus.PENDING for v in self._eval_status_dict[ix1].ix_match_status.values())

        if not has_pending:
            self._eval_status_dict[ix1].is_done = True
            self._eval_status_dict[ix1].owner = None
            conn.aux.pop('gen')
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)

            interpolated_ratings = self._evaluator.interpolate_ratings()
            test_iagents = [ia for ia in self._evaluator.indexed_agents if ia.role == AgentRole.TEST]
            with self._evaluator._db.db_lock:
                self._evaluator._db.commit_ratings(test_iagents, interpolated_ratings)

    def _handle_weights_request(self, msg: JsonDict, conn: ClientConnection):
        gen = msg['gen']
        thread = threading.Thread(target=self._manage_worker, args=(gen, conn),
                                  daemon=True, name=f'manage-eval-worker')
        thread.start()

    def _manage_worker(self, gen: Generation, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id

            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            self._pause(conn)
            self._update_weights(gen, conn)

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

    def _pause(self, conn: ClientConnection):
        logger.debug('Pausing %s...', conn)
        data = {
            'type': 'pause',
        }
        conn.aux['pending_pause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_pause_ack' not in conn.aux)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)
        data = {
            'type': 'unpause',
        }
        conn.aux['pending_unpause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_unpause_ack' not in conn.aux)

        logger.debug('Unpause of %s complete!', conn)

    def _handle_pause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

    def _update_weights(self, gen: Generation, conn: ClientConnection):
        self._controller.broadcast_weights(conn, gen)
        conn.aux['gen'] = gen


