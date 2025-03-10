from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import ClientConnection, Generation, EvalTag, ServerStatus, ClientId
from alphazero.logic.arena import Arena
from alphazero.logic.evaluator import Evaluator
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger
from util.py_util import find_largest_gap
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util

import threading
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = get_logger()


@dataclass
class EvalStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
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
        self._eval_status_dict: Dict[Generation, EvalStatus] = {}

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


    def _handle_server_disconnect(self, conn: ClientConnection):
        gen = conn.aux.pop('gen', None)
        if gen is not None:
            with self._lock:
                eval_status = self._eval_status_dict.get(gen, None)
                if eval_status is not None:
                    eval_status.owner = None

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
                lambda: len(self._mcts_evaluator._evaluated_gens) < self._controller.latest_gen())

    def _get_rating_data(self, conn: ClientConnection, gen: Generation) -> RatingData:
      #TODO: may not need this or need to modify
        with self._lock:
            rating_data = self._rating_data_dict.get(gen, None)
            if rating_data is None:
                rating_data = RatingData(gen, self._min_ref_strength, self._max_ref_strength)
                rating_data.est_rating = self._estimate_rating(gen)
                self._rating_data_dict[gen] = rating_data
                self._set_priority()

            rating_data.owner = conn.client_id
            return rating_data

    def _send_match_request(self, conn: ClientConnection):
        gen = conn.aux.get('gen', None)
        if gen is None:
            gen = self._mcts_evaluator.get_next_gen_to_eval(self._controller.params.target_rating_rate)
            assert gen is not None
            conn.aux['gen'] = gen

        with self._lock:
            if gen not in self._eval_status_dict:
                self._eval_status_dict[gen] = EvalStatus(mcts_gen=gen,
                                                        owner=conn.client_id,
                                                        is_done=False)
                self._set_priority()

        init_rating_estimate = self._mcts_evaluator.estimate_rating_nearby_gens(gen)
        next_match = self._mcts_evaluator.get_next_match_to_eval(gen,
                                                                 init_rating_estimate = init_rating_estimate)
        if next_match is None:
            self._eval_status_dict[gen].is_done = True
            logger.info("Inside of _send_match_request(): no more next match. Mark gen as done.")

        data = {
            'type': 'match-request',
            'agent1': {
                'gen': next_match.agent1.gen,
                'n_iters': next_match.agent1.n_iters,
                'set_temp_zero': next_match.agent1.set_temp_zero,
                'tag': next_match.agent1.tag,
            },
            'agent2': {
                'gen': next_match.agent2.gen,
                'n_iters': next_match.agent2.n_iters,
                'set_temp_zero': next_match.agent2.set_temp_zero,
                'tag': next_match.agent2.tag,
            },
            'n_games': next_match.n_games,
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
      #TODO: main change
        mcts_gen = msg['mcts_gen']
        ref_strength = msg['ref_strength']
        record = WinLossDrawCounts.from_json(msg['record'])

        rating = None
        with self._lock:
            rating_data = self._rating_data_dict[mcts_gen]
            assert rating_data.owner == conn.client_id
            assert conn.aux.get('gen', None) == mcts_gen
            rating_data.add_result(ref_strength, record)

            updated_record = rating_data.match_data[ref_strength]
            rating = rating_data.rating
            if rating is not None:
                conn.aux.pop('gen')
                rating_data.owner = None

        with self._controller.ratings_db_conn_pool.db_lock:
            self._commit_counts(mcts_gen, ref_strength, updated_record)
            if rating is not None:
                self._commit_rating(mcts_gen, rating)

        if rating is not None:
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)

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


