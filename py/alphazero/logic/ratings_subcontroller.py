from alphazero.logic.aux_subcontroller import AuxSubcontroller
from alphazero.logic import constants
from alphazero.logic.custom_types import ClientData, ClientId, Generation, NewModelSubscriber
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.loop_control_data import LoopControlData
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger
from util.py_util import find_largest_gap, get_neighboring_ints_sorted_by_proximity
from util.socket_util import recv_json, send_json
from util.sqlite3_util import ConnectionPool

from collections import defaultdict
from dataclasses import dataclass
import math
import threading
from typing import Dict, Optional


logger = get_logger()


"""
For now, hard-coding these constants.

If we want to make the configurable in the future, we may want the ability to configure them for a
running loop controller without requiring a restart. That would require some additional
infrastructure.
"""
N_GAMES = 100
N_MCTS_ITERS = 1600


class RatingData:
    """
    A RatingData summarizes match results for a given MCTS generation.
    """

    def __init__(self, mcts_gen: int, min_ref_strength: int, max_ref_strength: int):
        self.mcts_gen = mcts_gen
        self.min_ref_strength = min_ref_strength
        self.max_ref_strength = max_ref_strength

        self.est_rating = None
        self.rating_lower_bound = min_ref_strength - 1
        self.rating_upper_bound = max_ref_strength + 1

        self.owner: Optional[ClientId] = None  # client that is currently evaluating this gen

        self.match_data = defaultdict(WinLossDrawCounts)  # ref_strength -> mcts WLD
        self.rating = None

    def __str__(self):
        return (f'RatingData(mcts_gen={self.mcts_gen}, rating={self.rating}, '
                f'owner={self.owner}, est_rating={self.est_rating}, '
                f'rating_bounds=({self.rating_lower_bound}, {self.rating_upper_bound}), '
                f'match_data={dict(self.match_data)})')

    def __repr__(self):
        return str(self)

    def filtered_match_data(self) -> Dict[int, WinLossDrawCounts]:
        """
        Returns match data for ref strengths that have been tested at least N_GAMES times.

        With the current implementation, this should typically just return a copy of
        self.match_data. The only case where it could be different is if N_GAMES is changed
        between runs.
        """
        return {k: v for k, v in self.match_data.items() if v.n_games >= N_GAMES}

    def add_result(self, ref_strength: int, counts: WinLossDrawCounts, set_rating=True):
        self.match_data[ref_strength] += counts
        counts = self.match_data[ref_strength]

        if counts.n_games < N_GAMES:
            return

        if counts.win_rate() < 0.5:
            self.rating_upper_bound = min(self.rating_upper_bound, ref_strength)
        else:
            self.rating_lower_bound = max(self.rating_lower_bound, ref_strength)

        if set_rating:
            self.set_rating()

    def set_rating(self):
        if self.rating is not None:
            return

        if self.rating_upper_bound == self.min_ref_strength:
            self.rating = self.min_ref_strength
            return

        if self.rating_lower_bound == self.max_ref_strength:
            self.rating = self.max_ref_strength
            return

        if self.rating_lower_bound + 1 < self.rating_upper_bound:
            return

        assert self.rating_lower_bound + 1 == self.rating_upper_bound

        self.rating = self._interpolate_bounds(self.filtered_match_data())

    def _interpolate_bounds(self, match_data):
        """
        Interpolates between the lower and upper bound to estimate the critical strength.

        The interpolation formula is:

        rating = midpoint + spread_factor * adjustment

        where:

        midpoint = 0.5 * (x1 + x2)
        spread_factor = sqrt(x2 - x1)
        adjustment = 0.5 * (w1 - w2) / (w1 + w2)

        x1 = rating_lower_bound
        x2 = rating_upper_bound
        w1 = (win rate at x1) - 0.5
        w2 = 0.5 - (win rate at x2)

        If lower + 1 == upper, then this estimate is exactly the critical strength, and so serves
        as the exact rating.
        """
        lower_counts = match_data.get(self.rating_lower_bound, WinLossDrawCounts(win=1))
        upper_counts = match_data.get(self.rating_upper_bound, WinLossDrawCounts(loss=1))

        x1 = self.rating_lower_bound
        x2 = self.rating_upper_bound
        w1 = lower_counts.win_rate() - 0.5
        w2 = 0.5 - upper_counts.win_rate()

        assert x2 >= x1 + 1, (x1, x2, match_data)
        assert w1 >= 0, (w1, x1, match_data)
        assert w2 > 0, (w2, x2, match_data)

        midpoint = 0.5 * (x1 + x2)
        spread_factor = math.sqrt(x2 - x1)
        adjustment = 0.5 * (w1 - w2) / (w1 + w2)
        strength = midpoint + spread_factor * adjustment

        logger.debug(f'Interpolating bounds for gen={self.mcts_gen} match_data={match_data}: '
                     f'({x1}: {w1}, {x2}: {w2}) -> {midpoint} + {spread_factor} * {adjustment} = {strength}')

        assert x1 <= strength <= x2, (x1, strength, x2, match_data)
        return strength

    def get_next_strength_to_test(self):
        if self.rating is not None:
            return None

        match_data = self.filtered_match_data()

        if self.est_rating is not None:
            if self.rating_lower_bound < self.est_rating < self.rating_upper_bound:
                candidates = get_neighboring_ints_sorted_by_proximity(self.est_rating)
                for c in candidates:
                    if c not in match_data:
                        return c
                raise Exception(f'Unexpected state: {self}')

        est_rating = self._interpolate_bounds(match_data)
        candidates = get_neighboring_ints_sorted_by_proximity(est_rating)
        for c in candidates:
            if c not in match_data:
                return c
        raise Exception(f'Unexpected state: {self}')


RatingDataDict = Dict[Generation, RatingData]


class RatingsSubcontroller(NewModelSubscriber):
    """
    Used by the LoopController to manage ratings games. The actual games are played in external
    servers; this subcontroller acts as a sort of remote-manager of those servers.
    """

    def __init__(self, aux_controller: AuxSubcontroller):
        self.aux_controller = aux_controller
        self.aux_controller.subscribe_to_new_model_announcements(self)

        self.min_ref_strength = self.data.game_spec.reference_player_family.min_strength
        self.max_ref_strength = self.data.game_spec.reference_player_family.max_strength

        self.ratings_db_conn_pool = ConnectionPool(
            self.organizer.ratings_db_filename, constants.RATINGS_TABLE_CREATE_CMDS)

        self._started = False
        self._lock = threading.Lock()
        self._new_work_available = threading.Condition(self._lock)
        self._rating_data_dict: RatingDataDict = {}
        self._owner_dict: Dict[ClientId, RatingData] = {}

    def handle_new_model(self, generation: Generation):
        with self._lock:
            self._new_work_available.notify_all()

    @property
    def data(self) -> LoopControlData:
        return self.aux_controller.data

    @property
    def organizer(self) -> DirectoryOrganizer:
        return self.data.organizer

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True

            # Perform this in a separate thread because we want to use
            # ConnectionPool.close_connections() without interfering with other subcontrollers.
            thread = threading.Thread(target=self._start_helper, name='ratings-start', daemon=True)
            thread.start()
            thread.join()

    def _start_helper(self):
        self.dump_x_var_data()
        self.load_past_data()

    def dump_x_var_data(self):
        """
        Transfer self-play metadata from the self-play db to the ratings db.
        """
        logger.info(f'Dumping ratings x-data...')
        conn = self.ratings_db_conn_pool.get_connection()
        c = conn.cursor()

        mcts_gen_set = set()
        res = c.execute('SELECT mcts_gen FROM x_values')
        for gen in res.fetchall():
            mcts_gen_set.add(gen[0])

        latest_gen = self.organizer.get_latest_generation()
        full_mcts_gen_set = set(range(1, latest_gen + 1))

        missing_mcts_gen_set = full_mcts_gen_set - mcts_gen_set
        if not missing_mcts_gen_set:
            return
        missing_mcts_gen_list = list(missing_mcts_gen_set)

        placeholders = ', '.join('?' * len(missing_mcts_gen_list))
        query1 = ('SELECT gen, positions_evaluated, batches_evaluated, games FROM self_play_metadata '
                  'WHERE gen IN (%s)' % placeholders)
        query2 = ('SELECT gen, start_timestamp, end_timestamp FROM timestamps '
                  'WHERE gen IN (%s)' % placeholders)

        with self.data.self_play_db_conn_pool.db_lock:
            training_db_conn = self.data.self_play_db_conn_pool.get_connection()

            c2 = training_db_conn.cursor()
            c2.execute(query1, missing_mcts_gen_list)
            results1 = list(c2.fetchall())

            c2.execute(query2, missing_mcts_gen_list)
            results2 = list(c2.fetchall())

            self.data.self_play_db_conn_pool.close_connections()

        runtime_dict = defaultdict(int)
        for gen, start_timestamp, end_timestamp in results2:
            runtime_dict[gen] += end_timestamp - start_timestamp

        values = []
        for gen, n_evaluated_positions, n_batches_evaluated, n_games in results1:
            runtime = runtime_dict[gen]
            x_value_tuple = (gen, n_games, runtime, n_evaluated_positions, n_batches_evaluated)
            values.append(x_value_tuple)

        c.executemany('INSERT INTO x_values VALUES (?, ?, ?, ?, ?)', values)
        conn.commit()

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

    def load_past_data(self):
        logger.info(f'Loading past ratings data...')
        conn = self.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE mcts_iters = ?',
                        (N_MCTS_ITERS,))

        for mcts_gen, ref_strength, mcts_wins, draws, ref_wins in res.fetchall():
            if mcts_gen not in self._rating_data_dict:
                data = RatingData(mcts_gen, self.min_ref_strength, self.max_ref_strength)
                self._rating_data_dict[mcts_gen] = data
            counts = WinLossDrawCounts(mcts_wins, ref_wins, draws)
            self._rating_data_dict[mcts_gen].add_result(ref_strength, counts, set_rating=False)

        for data in self._rating_data_dict.values():
            data.set_rating()

        for gen, data in self._rating_data_dict.items():
            if data.rating is None:
                data.est_rating = self._estimate_rating(gen)

    def add_ratings_manager(self, client_data: ClientData):
        reply = {
            'type': 'handshake_ack',
            'client_id': client_data.client_id,
        }
        self.aux_controller.add_asset_metadata_to_reply(reply)
        send_json(client_data.sock, reply)

        self.start()
        logger.info(f'Starting ratings-recv-loop for {client_data}...')
        threading.Thread(target=self.recv_loop, name='ratings-recv-loop',
                         args=(client_data,), daemon=True).start()

    def send_match_request(self, client_data: ClientData):
        with self._lock:
            rating_data = self._owner_dict.get(client_data.client_id, None)
        if rating_data is None:
            gen = self._get_next_gen_to_rate()
            with self._lock:
                rating_data = self._rating_data_dict.get(gen, None)
                if rating_data is None:
                    rating_data = RatingData(gen, self.min_ref_strength, self.max_ref_strength)
                    rating_data.est_rating = self._estimate_rating(gen)
                    self._rating_data_dict[gen] = rating_data

                rating_data.owner = client_data.client_id
                self._owner_dict[client_data.client_id] = rating_data

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
        send_json(client_data.sock, data)

    def handle_disconnect(self, client_data: ClientData):
        self.aux_controller.handle_disconnect(client_data)
        with self._lock:
            rating_data = self._owner_dict.pop(client_data.client_id, None)
            if rating_data is not None:
                rating_data.owner = None

    def recv_loop(self, client_data: ClientData):
        try:
            while True:
                try:
                    msg = recv_json(client_data.sock)
                except OSError:
                    self.handle_disconnect(client_data)
                    return

                msg_type = msg['type']
                if msg_type == 'asset_request':
                    self.aux_controller.send_asset(msg['asset'], client_data)
                elif msg_type == 'work-request':
                    self.send_match_request(client_data)
                elif msg_type == 'match-result':
                    self.handle_match_result(msg, client_data)
        except:
            logger.error(
                f'Unexpected error in RatingsSubcontroller.recv_loop({client_data}):',
                exc_info=True)
            self.data.signal_error()

    def commit_rating(self, gen: int, rating: float):
        rating_tuple = (gen, N_MCTS_ITERS, N_GAMES, rating)

        conn = self.ratings_db_conn_pool.get_connection()
        c = conn.cursor()
        c.execute('REPLACE INTO ratings VALUES (?, ?, ?, ?)', rating_tuple)
        conn.commit()

    def commit_counts(self, mcts_gen: int, ref_strength: int, counts: WinLossDrawCounts):
        conn = self.ratings_db_conn_pool.get_connection()
        match_tuple = (mcts_gen, N_MCTS_ITERS, ref_strength,
                       counts.win, counts.draw, counts.loss)
        c = conn.cursor()
        c.execute('REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?)', match_tuple)
        conn.commit()

    def handle_match_result(self, msg, client_data: ClientData):
        mcts_gen = msg['mcts_gen']
        ref_strength = msg['ref_strength']
        record = WinLossDrawCounts.from_json(msg['record'])

        rating = None
        with self._lock:
            rating_data = self._rating_data_dict[mcts_gen]
            assert rating_data.owner == client_data.client_id
            assert self._owner_dict[client_data.client_id] is rating_data
            rating_data.add_result(ref_strength, record)

            updated_record = rating_data.match_data[ref_strength]
            rating = rating_data.rating
            if rating is not None:
                self._owner_dict.pop(client_data.client_id)
                rating_data.owner = None

        with self.ratings_db_conn_pool.db_lock:
            self.commit_counts(mcts_gen, ref_strength, updated_record)
            if rating is not None:
                self.commit_rating(mcts_gen, rating)

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
        latest_gen = self.organizer.get_latest_generation()
        next_gen = None
        if latest_gen > 0:
            next_gen = self._get_next_gen_to_rate_helper(latest_gen)

        if next_gen is None:
            with self._lock:
                self._new_work_available.wait()
            next_gen = self._get_next_gen_to_rate()

        assert next_gen is not None
        return next_gen

    def _get_next_gen_to_rate_helper(self, latest_gen: Generation) -> Optional[Generation]:
        """
        Helper to _get_next_gen_to_rate(). Assumes that latest_gen > 0.
        """
        logger.debug(f'Getting next gen to rate, latest_gen={latest_gen}...')
        with self._lock:
            taken_gens = [g for g, r in self._rating_data_dict.items() if r.rating is not None or r.owner is not None]
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

        n_taken_gens = len(taken_gens)
        if n_taken_gens < 2:
            assert latest_gen < 2
            logger.debug(f'Waiting for new model (latest={latest_gen}, n_taken:{n_taken_gens})...')
            return None

        left, right = find_largest_gap(taken_gens)
        gap = right - left
        if 2 * latest_gap >= gap:
            logger.debug(
                f'Large gap to latest, rating latest '
                f'(biggest-gap:[{left}, {right}], latest={latest_gen})...')
            return latest_gen
        if max(gap, latest_gap) == 1:
            logger.debug(f'Waiting for new model, all rated (latest={latest_gen})...')
            return None
        if left + 1 == right:
            if latest_gen > right:
                logger.debug(f'No existing gaps, rating latest ({latest_gen})...')
                return latest_gen
            logger.debug(f'Waiting for new model (latest={latest_gen}, n_taken:{n_taken_gens})...')
            return None

        mid = (left + right) // 2
        logger.debug(f'Rating gen {mid} (biggest-gap:[{left}, {right}], latest={latest_gen})...')
        return mid
