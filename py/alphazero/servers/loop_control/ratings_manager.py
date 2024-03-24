from .loop_controller_interface import LoopControllerInterface
from .rating_data import N_GAMES, N_MCTS_ITERS, RatingData, RatingDataDict

from alphazero.logic.custom_types import ClientConnection, ClientId, Generation
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger
from util.py_util import find_largest_gap
from util.socket_util import JsonDict

import logging
import threading
from typing import Dict, Optional


logger = get_logger()


class RatingsManager:
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller

        self._min_ref_strength = controller.game_spec.reference_player_family.min_strength
        self._max_ref_strength = controller.game_spec.reference_player_family.max_strength

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cv = threading.Condition(self._lock)
        self._rating_data_dict: RatingDataDict = {}
        self._owner_dict: Dict[ClientId, RatingData] = {}

    def add_server(self, conn: ClientConnection):
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
        }
        conn.socket.send_json(reply)

        self._start()
        logger.info(f'Starting ratings-recv-loop for {conn}...')
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'ratings-server',
            disconnect_handler=self._handle_server_disconnect)

    def add_worker(self, conn: ClientConnection):
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'ratings-worker')

    def notify_of_new_model(self):
        """
        Notify manager that there is new work to do.
        """
        with self._lock:
            self._new_work_cv.notify_all()

    def _start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._load_past_data()

    def _load_past_data(self):
        logger.info(f'Loading past ratings data...')
        conn = self._controller.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE mcts_iters = ?',
                        (N_MCTS_ITERS,))

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

    def _handle_server_disconnect(self, conn: ClientConnection):
        with self._lock:
            rating_data = self._owner_dict.pop(conn.client_id, None)
            if rating_data is not None:
                rating_data.owner = None

    def _server_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        if msg_type != 'log' and logger.isEnabledFor(logging.DEBUG):
            # no need to double-log log-messages
            logger.debug(f'ratings-server received json message: {msg}')

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'worker-exit':
            self._controller.handle_worker_exit(msg, conn)
        elif msg_type == 'work-request':
            self._handle_work_request(conn)
        elif msg_type == 'match-result':
            self._handle_match_result(msg, conn)
        else:
            logger.warn(f'ratings-server: unknown message type: {msg}')
        return False

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'ratings-worker received json message: {msg}')

        if msg_type == 'log':
            self._controller.handle_log_msg(msg, conn)
        elif msg_type == 'pause-ack':
            self._controller.handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._controller.handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warn(f'ratings-worker: unknown message type: {msg}')
        return False

    def _handle_work_request(self, conn: ClientConnection):
        with self._lock:
            rating_data = self._owner_dict.get(conn.client_id, None)
        if rating_data is None:
            gen = self._get_next_gen_to_rate()
            with self._lock:
                rating_data = self._rating_data_dict.get(gen, None)
                if rating_data is None:
                    rating_data = RatingData(gen, self._min_ref_strength, self._max_ref_strength)
                    rating_data.est_rating = self._estimate_rating(gen)
                    self._rating_data_dict[gen] = rating_data

                rating_data.owner = conn.client_id
                self._owner_dict[conn.client_id] = rating_data

        assert rating_data.rating is None
        strength = rating_data.get_next_strength_to_test()
        assert strength is not None
        data = {
            'type': 'match-request',
            'mcts_gen': rating_data.mcts_gen,
            'ref_strength': strength,
            'n_games': N_GAMES,
            'n_mcts_iters': N_MCTS_ITERS,
        }
        conn.socket.send_json(data)

    def _handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        mcts_gen = msg['mcts_gen']
        ref_strength = msg['ref_strength']
        record = WinLossDrawCounts.from_json(msg['record'])

        rating = None
        with self._lock:
            rating_data = self._rating_data_dict[mcts_gen]
            assert rating_data.owner == conn.client_id
            assert self._owner_dict[conn.client_id] is rating_data
            rating_data.add_result(ref_strength, record)

            updated_record = rating_data.match_data[ref_strength]
            rating = rating_data.rating
            if rating is not None:
                self._owner_dict.pop(conn.client_id)
                rating_data.owner = None

        with self._controller.ratings_db_conn_pool.db_lock:
            self._commit_counts(mcts_gen, ref_strength, updated_record)
            if rating is not None:
                self._commit_rating(mcts_gen, rating)

        if rating is not None:
            self._controller.notify_of_new_rating()

    def _handle_weights_request(self, msg: JsonDict, conn: ClientConnection):
        gen = msg['gen']
        self._controller.start_worker(conn, gen)

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
        rating_tuple = (gen, N_MCTS_ITERS, N_GAMES, rating)

        conn = self._controller.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('REPLACE INTO ratings VALUES (?, ?, ?, ?)', rating_tuple)
        conn.commit()

    def _commit_counts(self, mcts_gen: Generation, ref_strength: int, counts: WinLossDrawCounts):
        """
        Assumes that ratings_db_conn_pool.db_lock is held.
        """
        conn = self._controller.ratings_db_conn_pool.get_connection()
        match_tuple = (mcts_gen, N_MCTS_ITERS, ref_strength, counts.win, counts.draw, counts.loss)
        c = conn.cursor()
        c.execute('REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def _get_next_gen_to_rate(self) -> Generation:
        """
        Returns the next generation to rate

        Description of selection algorithm:

        Let G be the set of gens that we have graded or are currently graded thus far, and let M be
        the max generation that exists in the models directory.

        If M is at least 10 greater than the max element of G, then we return M. This is to keep up
        with a currently running alphazero run.

        Otherwise, if 1 is not in G, then we return 1.

        Finally, we find the largest gap in G, and return the midpoint of that gap. If G is fully
        saturated, we return M if M is not in G. If no such number exists, we return None.
        """
        latest_gen = self._controller.organizer.get_latest_model_generation()
        next_gen = None
        if latest_gen > 0:
            next_gen = self._get_next_gen_to_rate_helper(latest_gen)

        if next_gen is None:
            with self._lock:
                self._new_work_cv.wait()
            next_gen = self._get_next_gen_to_rate()

        assert next_gen is not None
        return next_gen

    def _get_next_gen_to_rate_helper(self, latest_gen: Generation) -> Optional[Generation]:
        """
        Helper to _get_next_gen_to_rate(). Assumes that latest_gen > 0.
        """
        logger.debug(f'Getting next gen to rate, latest_gen={latest_gen}...')
        with self._lock:
            taken_gens = [g for g, r in self._rating_data_dict.items(
            ) if r.rating is not None or r.owner is not None]
            taken_gens.sort()
        if not taken_gens:
            logger.debug(f'No gens yet rated, rating latest ({latest_gen})...')
            return latest_gen

        max_taken_gen = taken_gens[-1]

        assert latest_gen >= max_taken_gen
        latest_gap = latest_gen - max_taken_gen

        latest_gap_threshold = 10
        if latest_gap >= latest_gap_threshold:
            logger.debug(f'{latest_gap_threshold}+ gap to latest, rating latest ({latest_gen})...')
            return latest_gen

        if taken_gens[0] > 1:
            logger.debug(f'Gen-1 not yet rated, rating it...')
            return 1

        if latest_gen == 1:
            logger.debug(f'Waiting for new model (latest={latest_gen})...')
            return None

        if len(taken_gens) == 1:
            logger.debug(f'No existing gaps, rating latest ({latest_gen})...')
            return latest_gen

        left, right = find_largest_gap(taken_gens)
        gap = right - left
        if 2 * latest_gap >= gap:
            logger.debug(
                f'Large gap to latest, rating latest={latest_gen} '
                f'(biggest-gap:[{left}, {right}], latest-gap:[{max_taken_gen}, {latest_gen}])...')
            return latest_gen
        if max(gap, latest_gap) == 1:
            logger.debug(f'Waiting for new model, all rated (latest={latest_gen})...')
            return None
        if left + 1 == right:
            if latest_gen > right:
                logger.debug(f'No existing gaps, rating latest ({latest_gen})...')
                return latest_gen
            logger.debug(f'Waiting for new model (latest={latest_gen})...')
            return None

        mid = (left + right) // 2
        logger.debug(f'Rating gen {mid} (biggest-gap:[{left}, {right}], latest={latest_gen})...')
        return mid
