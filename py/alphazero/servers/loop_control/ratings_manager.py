from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface
from .rating_data import N_GAMES, RatingData, RatingDataDict

from alphazero.logic.custom_types import ClientConnection, Generation, RatingTag
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger
from util.py_util import find_largest_gap
from util.socket_util import JsonDict, SocketSendException

from enum import Enum
import logging
import threading
from typing import Optional


logger = get_logger()


class ServerStatus(Enum):
    DISCONNECTED = 'disconnected'
    BLOCKED = 'blocked'
    READY = 'ready'


class RatingsManager:
    """
    A separate RatingsManager is created for each rating-tag.
    """
    def __init__(self, controller: LoopControllerInterface, tag: RatingTag):
        self._tag = tag
        self._controller = controller

        self._min_ref_strength = controller.game_spec.reference_player_family.min_strength
        self._max_ref_strength = controller.game_spec.reference_player_family.max_strength

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cond = threading.Condition(self._lock)
        self._rating_data_dict: RatingDataDict = {}

    def add_server(self, conn: ClientConnection):
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
        }
        conn.socket.send_json(reply)

        conn.aux['status_cond'] = threading.Condition()
        conn.aux['status'] = ServerStatus.BLOCKED

        self._start()
        logger.info('Starting ratings-recv-loop for %s...', conn)
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'ratings-server',
            disconnect_handler=self._handle_server_disconnect)

        thread = threading.Thread(target=self._manage_server, args=(conn,),
                                  daemon=True, name=f'manage-self-play-server')
        thread.start()

    def add_worker(self, conn: ClientConnection):
        conn.aux['ack_cond'] = threading.Condition()

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'ratings-worker',
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
        dict_len = len(self._rating_data_dict)
        rating_in_progress = any(r.rating is None for r in self._rating_data_dict.values())

        target_rate = self._controller.params.target_rating_rate
        num = dict_len + (0 if rating_in_progress else 1)
        den = max(1, latest_gen)
        current_rate = num / den

        elevate = current_rate < target_rate
        logger.debug('Ratings elevate-priority:%s (latest=%s, dict_len=%s, in_progress=%s, '
                     'current=%.2f, target=%.2f)', elevate, latest_gen, dict_len,
                     rating_in_progress, current_rate, target_rate)
        self._controller.set_ratings_priority(elevate)

    def _start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._load_past_data()

    def _load_past_data(self):
        logger.info('Loading past ratings data...')
        conn = self._controller.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE tag = ?',
                        (self._tag,))

        for mcts_gen, ref_strength, mcts_wins, draws, ref_wins in res.fetchall():
            if mcts_gen not in self._rating_data_dict:
                data = RatingData(mcts_gen, self._min_ref_strength, self._max_ref_strength)
                self._rating_data_dict[mcts_gen] = data
            counts = WinLossDrawCounts(mcts_wins, ref_wins, draws)
            self._rating_data_dict[mcts_gen].add_result(ref_strength, counts, set_rating=False)

        for data in self._rating_data_dict.values():
            data.set_rating()

        for gen, data in self._rating_data_dict.items():
            if data.rating is None:
                data.est_rating = self._estimate_rating(gen)

        self._set_priority()

    def _handle_server_disconnect(self, conn: ClientConnection):
        gen = conn.aux.pop('gen', None)
        if gen is not None:
            with self._lock:
                rating_data = self._rating_data_dict.get(gen, None)
                if rating_data is not None:
                    rating_data.owner = None

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
                lambda: len(self._rating_data_dict) < self._controller.latest_gen())

    def _get_rating_data(self, conn: ClientConnection, gen: Generation) -> RatingData:
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
            gen = self._get_next_gen_to_rate()
            conn.aux['gen'] = gen

        rating_data = self._get_rating_data(conn, gen)
        assert rating_data.rating is None
        strength = rating_data.get_next_strength_to_test()
        assert strength is not None

        data = {
            'type': 'match-request',
            'mcts_gen': rating_data.mcts_gen,
            'ref_strength': strength,
            'n_games': N_GAMES,
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
        if msg_type != 'log':
            # no need to double-log log-messages
            logger.debug('ratings-server received json message: %s', msg)

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'worker-exit':
            self._controller.handle_worker_exit(msg, conn)
        elif msg_type == 'ready':
            self._handle_ready(conn)
        elif msg_type == 'match-result':
            self._handle_match_result(msg, conn)
        else:
            logger.warning('ratings-server: unknown message type: %s', msg)
        return False

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('ratings-worker received json message: %s', msg)

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('ratings-worker: unknown message type: %s', msg)
        return False

    def _handle_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.READY
            status_cond.notify_all()

    def _handle_match_result(self, msg: JsonDict, conn: ClientConnection):
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
                                  daemon=True, name=f'manage-ratings-worker')
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

    def _estimate_rating(self, gen: Generation) -> Optional[float]:
        """
        Estimates the rating for a given MCTS generation by interpolating nearby generations.

        Caller is responsible for holding self._lock if necessary.
        """
        rated_gens = [g for g, r in self._rating_data_dict.items() if r.rating is not None]

        left = max([g for g in rated_gens if g < gen], default=None)
        right = min([g for g in rated_gens if g > gen], default=None)

        left_rating = None if left is None else self._rating_data_dict[left].rating
        right_rating = None if right is None else self._rating_data_dict[right].rating

        if left is None:
            return right_rating
        if right is None:
            return left_rating

        left_weight = right - gen
        right_weight = gen - left

        num = left_weight * left_rating + right_weight * right_rating
        den = left_weight + right_weight
        return num / den

    def _commit_rating(self, gen: Generation, rating: float):
        """
        Assumes that ratings_db_conn_pool.db_lock is held.
        """
        rating_tuple = (self._tag, gen, N_GAMES, rating)

        conn = self._controller.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('REPLACE INTO ratings VALUES (?, ?, ?, ?)', rating_tuple)
        conn.commit()

    def _commit_counts(self, mcts_gen: Generation, ref_strength: int, counts: WinLossDrawCounts):
        """
        Assumes that ratings_db_conn_pool.db_lock is held.
        """
        conn = self._controller.ratings_db_conn_pool.get_connection()
        match_tuple = (self._tag, mcts_gen, ref_strength, counts.win, counts.draw, counts.loss)
        c = conn.cursor()
        c.execute('REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def _get_next_gen_to_rate(self) -> Generation:
        """
        Returns the next generation to rate. Assumes that there is at least one generation that has
        not been rated and is not currently being rated.

        Description of selection algorithm:

        Let G be the set of gens that we have graded or are currently graded thus far, and let M be
        the max generation that exists in the models directory.

        If M is at least 10 greater than the max element of G, then we return M. This is to keep up
        with a currently running alphazero run.

        Otherwise, if 1 is not in G, then we return 1.

        Finally, we find the largest gap in G, and return the midpoint of that gap. If G is fully
        saturated, we return M, which cannot be in G due to the above assumption.
        """
        latest_gen = self._controller.latest_gen()
        assert latest_gen > 0, latest_gen

        logger.debug('Getting next gen to rate, latest_gen=%s...', latest_gen)
        with self._lock:
            taken_gens = [g for g, r in self._rating_data_dict.items()
                          if r.rating is not None or r.owner is not None]
            taken_gens.sort()
        if not taken_gens:
            logger.debug('No gens yet rated, rating latest (%s)...', latest_gen)
            return latest_gen

        max_taken_gen = taken_gens[-1]

        assert latest_gen >= max_taken_gen
        latest_gap = latest_gen - max_taken_gen

        latest_gap_threshold = 10
        if latest_gap >= latest_gap_threshold:
            logger.debug('%s+ gap to latest, rating latest (%s)...', latest_gap_threshold,
                         latest_gen)
            return latest_gen

        if taken_gens[0] > 1:
            logger.debug('Gen-1 not yet rated, rating it...')
            return 1

        assert latest_gen != 1, latest_gen

        if len(taken_gens) == 1:
            logger.debug('No existing gaps, rating latest (%s)...', latest_gen)
            return latest_gen

        left, right = find_largest_gap(taken_gens)
        gap = right - left
        if 2 * latest_gap >= gap:
            logger.debug(
                'Large gap to latest, rating latest=%s '
                '(biggest-gap:[%s, %s], latest-gap:[%s, %s])...',
                latest_gen, left, right, max_taken_gen, latest_gap)
            return latest_gen

        assert max(gap, latest_gap) > 1, (gap, latest_gap)

        if left + 1 == right:
            assert latest_gen > right, (latest_gen, right)
            logger.debug('No existing gaps, rating latest (%s)...', latest_gen)
            return latest_gen

        mid = (left + right) // 2
        logger.debug('Rating gen %s (biggest-gap:[%s, %s], latest=%s)...',
                     mid, left, right, latest_gen)
        return mid
