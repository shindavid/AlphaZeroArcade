from __future__ import annotations

from .gpu_contention_table import GpuContentionTable
from .rating_data import N_GAMES, RatingData, RatingDataDict

from alphazero.logic.custom_types import ClientConnection, Domain, FileToTransfer, Generation, \
    RatingTag, ServerStatus
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.servers.loop_control.gaming_manager_base import GamingManagerBase, ManagerConfig, \
    ServerAuxBase, WorkerAux
from util.py_util import find_largest_gap
from util.socket_util import JsonDict

from dataclasses import dataclass
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


@dataclass
class RatingsServerAux(ServerAuxBase):
    """
    Auxiliary data stored per server connection.
    """
    gen: Optional[Generation] = None

    def work_in_progress(self) -> bool:
        return self.gen is not None


class RatingsManager(GamingManagerBase):
    """
    A separate RatingsManager is created for each rating-tag.
    """
    def __init__(self, controller: LoopController, tag: RatingTag):
        manager_config = ManagerConfig(
            worker_aux_class=WorkerAux,
            server_aux_class=RatingsServerAux,
            server_name='ratings-server',
            worker_name='ratings-worker',
            domain=Domain.RATINGS,
        )
        super().__init__(controller, manager_config, tag=tag)
        self._min_ref_strength = controller.game_spec.reference_player_family.min_strength
        self._max_ref_strength = controller.game_spec.reference_player_family.max_strength
        self._rating_data_dict: RatingDataDict = {}

    def set_priority(self):
        dict_len = len(self._rating_data_dict)
        rating_in_progress = any(r.rating is None for r in self._rating_data_dict.values())
        self._set_domain_priority(dict_len, rating_in_progress)

    def load_past_data(self):
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

        self.set_priority()

    def num_evaluated_gens(self):
        return len(self._rating_data_dict)

    def _task_finished(self):
        rated_percent = self.num_evaluated_gens() / self._controller._organizer.get_latest_model_generation()
        return rated_percent >= self._controller.params.target_rating_rate

    def handle_server_disconnect(self, conn: ClientConnection):
        aux = conn.aux
        gen = aux.gen
        aux.gen = None
        if gen is not None:
            with self._lock:
                rating_data = self._rating_data_dict.get(gen, None)
                if rating_data is not None:
                    rating_data.owner = None

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

    def _get_rating_data(self, conn: ClientConnection, gen: Generation) -> RatingData:
        with self._lock:
            rating_data = self._rating_data_dict.get(gen, None)
            if rating_data is None:
                rating_data = RatingData(gen, self._min_ref_strength, self._max_ref_strength)
                rating_data.est_rating = self._estimate_rating(gen)
                self._rating_data_dict[gen] = rating_data
                self.set_priority()

            rating_data.owner = conn.client_id
            return rating_data

    def send_match_request(self, conn: ClientConnection):
        aux: RatingsManager.ServerAux = conn.aux
        gen = aux.gen
        if gen is None:
            gen = self._get_next_gen_to_rate()
            aux.gen = gen

        rating_data = self._get_rating_data(conn, gen)
        assert rating_data.rating is None
        strength = rating_data.get_next_strength_to_test()
        assert strength is not None

        game = self._controller._run_params.game
        tag = self._controller._run_params.tag
        eval_binary = FileToTransfer.from_src_scratch_path(
            source_path=self._controller.organizer_binary_path,
            scratch_path=f'bin/{game}',
            asset_path_mode='scratch'
        )
        files_required = [eval_binary]

        for dep in self._controller.game_spec.extra_runtime_deps:
            dep_binary = FileToTransfer.from_src_scratch_path(
                source_path=dep, scratch_path=dep, asset_path_mode='scratch'
            )
            files_required.append(dep_binary)

        model_file = None
        if rating_data.mcts_gen > 0:
            model_file = FileToTransfer.from_src_scratch_path(
                source_path=self._controller._organizer.get_model_filename(rating_data.mcts_gen),
                scratch_path=f'eval-models/{tag}/gen-{rating_data.mcts_gen}.pt',
                asset_path_mode='scratch')
            files_required.append(model_file)

        data = {
            'type': 'match-request',
            'mcts_agent': {
                'gen': rating_data.mcts_gen,
                'set_temp_zero': True,
                'tag': tag,
                'binary': eval_binary.scratch_path,
                'model': model_file.scratch_path if model_file else None,
                },
            'ref_strength': strength,
            'n_games': N_GAMES,
            'files_required': [f.to_dict() for f in files_required],
        }
        conn.socket.send_json(data)

    def handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        aux: RatingsManager.ServerAux = conn.aux

        mcts_gen = msg['mcts_gen']
        ref_strength = msg['ref_strength']
        record = WinLossDrawCounts.from_json(msg['record'])

        rating = None
        with self._lock:
            rating_data = self._rating_data_dict[mcts_gen]
            assert rating_data.owner == conn.client_id
            assert aux.gen == mcts_gen
            rating_data.add_result(ref_strength, record)

            updated_record = rating_data.match_data[ref_strength]
            rating = rating_data.rating
            if rating is not None:
                aux.gen = None
                rating_data.owner = None

        with self._controller.ratings_db_conn_pool.db_lock:
            self._commit_counts(mcts_gen, ref_strength, updated_record)
            if rating is not None:
                self._commit_rating(mcts_gen, rating)

        if rating is not None:
            table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
            table.release_lock(conn.client_domain)

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
