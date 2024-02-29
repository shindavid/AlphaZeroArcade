#!/usr/bin/env python3

"""
Measures the learning progress of an alphazero run.

This is accomplished by testing each generation produced by the run against a reference family of agents. This family
is specified on a per-game basis in games.py. For example, for othello, the family is the edax family of agents. The
important attribute of this family is that it is parameterized by a single integer, which we call the "strength". The
larger the strength, the stronger the agent. For a family F and a strength s, we let F[s] denote the agent in the
family with strength s.

For a given opponent P, we loosely say that P has a "rating" of r if P is equal in skill to F[r]. A more precise
definition is given below.

For the purpose of creating a learning curve, so that we can compare learning progress across runs, it is not
necessary to compute a rating at every single generation. This script thus instead samples generations at regular
intervals, and computes the rating of the sampled generations. If you let it run longer, it starts shrinking those
intervals to get a finer and finer-grained learning curve.

Note that there is a lot of noise in the rating computation - even if an agent's "true" strength is 7, if we don't
test for enough games, it might end up with a strength of, say, 7 +/- 2, due to randomness in the games. An EMA is a
good way to smooth out this noise, but that is left as a detail for the visualizing script.

*** MORE PRECISE DEFINITION OF RATING ***

The loose definition of rating offered above is nice, but details on how to interpolate between integer values are
needed.

Our current attempt to fill in those details are as follows:

For a player P, let r be the smallest integer such that P has a losing record against F[r]. This implies that P has a
non-losing record against F[r-1]. Let w(r) be the win-rate of P against F[r], and let w(r-1) be the win-rate of P
against F[r-1]. We must have w(r) < 0.5 <= w(r-1). Then we define the rating of P to be the weighted-average of
(r-1) and r, where the weights are proportional to the distances of w(r) and w(r-1) from 0.5.

More precisely, the rating of P is:

((r-1) * (0.5 - w(r)) + r * (w(r-1) - 0.5)) / (w(r-1) - w(r))

Some illustrative examples, using a plausible evolution of win rates as P progresses from F[7]-level to F[8]-level:

* If P wins 50% against F[7] and 0% against F[8], then the weights for 7 and 8 are 0.5-0=0.5 and 0.5-0.5=0,
respectively, resulting in a rating of 7*0.5 / 0.5 = 7

* If P wins 60% against F[7] and 0% against F[8], then the weights for 7 and 8 are 0.5-0=0.5 and 0.6-0.5=0.1,
respectively, resulting in a rating of (7*0.5 + 8*0.1) / (0.5 + 0.1) = 7.166...

* If P wins 70% against F[7] and 20% against F[8], then the weights for 7 and 8 are 0.5-0.2=0.3 and 0.7-0.5=0.2,
respectively, resulting in a rating of (7*0.3 + 8*0.2) / (0.3 + 0.2) = 7.4

* If P wins 90% against F[7] and 40% against F[8], then the weights for 7 and 8 are 0.5-0.4=0.1 and 0.9-0.5=0.4,
respectively, resulting in a rating of (7*0.1 + 8*0.4) / (0.1 + 0.4) = 7.8

This definition is not as sophisticated as a rating system with probabilistic interpretations like Bradley-Terry. But,
in a Bradley-Terry based system, a given agent's rating is dependent on the population of agents that we select, and
selecting populations in a principled way is a thorny theoretical problem. We hope to devise a general approach based
on Bradley-Terry in the future, but for now, we use the simpler definition above.
"""
import argparse
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List

import game_index
from alphazero.logic.common_params import CommonParams
from alphazero.logic.directory_organizer import DirectoryOrganizer
from alphazero.logic.ratings import extract_match_record, WinLossDrawCounts
from util import subprocess_util
from util.logging_util import configure_logger, get_logger
from util.sqlite3_util import open_readonly_conn


logger = get_logger()


class Params:
    alphazero_dir: str
    game: str
    tags: List[str]
    clear_db: bool
    n_games: int
    mcts_iters: int
    parallelism_factor: int
    num_search_threads: int
    binary: str
    daemon_mode: bool

    @staticmethod
    def load(args):
        assert args.tag, 'Required option: -t'
        Params.alphazero_dir = args.alphazero_dir
        Params.game = args.game
        Params.tags = [t for t in args.tag.split(',') if t]
        Params.clear_db = bool(args.clear_db)
        Params.n_games = args.n_games
        Params.mcts_iters = args.mcts_iters
        Params.parallelism_factor = args.parallelism_factor
        Params.num_search_threads = args.num_search_threads
        Params.binary = args.binary
        Params.daemon_mode = bool(args.daemon_mode)

    @staticmethod
    def add_args(parser):
        parser.add_argument('-C', '--clear-db', action='store_true', help='clear everything from database')
        parser.add_argument('-G', '--n-games', type=int, default=100,
                            help='number of games to play per matchup (default: %(default)s))')
        parser.add_argument('-i', '--mcts-iters', type=int, default=1600,
                            help='number of MCTS iterations per move (default: %(default)s)')
        parser.add_argument('-p', '--parallelism-factor', type=int, default=100,
                            help='parallelism factor (default: %(default)s)')
        parser.add_argument('-n', '--num-search-threads', type=int, default=4,
                            help='num search threads per game (default: %(default)s)')
        parser.add_argument('-b', '--binary', help='optional binary')
        parser.add_argument('-D', '--daemon-mode', action='store_true', help='daemon mode (run forever)')
        CommonParams.add_args(parser)


def load_args():
    parser = argparse.ArgumentParser()
    Params.add_args(parser)
    args = parser.parse_args()
    Params.load(args)


@dataclass
class WorkItem:
    """
    Represents a unit of work for the Arena to process. The key field is mcts_gen, which is the generation that needs
    to be rated.

    The est_rating field is used as a hint to guide the rating computation.

    The rating_gap field is the size of the generation-gap that would get filled by rating mcts_gen.

    The recency_boost field is True if the item is chosen because a new generation has been produced by a currently
    running alphazero process. The selection process gives a boost for recency because we usually want to see recent
    results in the skill visualization graph quickly.
    """
    mcts_gen: int
    est_rating: Optional[float]
    rating_gap: int
    recency_boost: bool
    arena: 'Arena'


class Arena:
    def __init__(self, common_params: CommonParams):
        tag = common_params.tag
        self.tag = tag
        self.game_spec = game_index.get_game_spec(Params.game)
        self.organizer = DirectoryOrganizer(common_params)

        self.min_ref_strength = self.game_spec.reference_player_family.min_strength
        self.max_ref_strength = self.game_spec.reference_player_family.max_strength

        # mcts_gen -> ref_strength -> WLD
        self.match_data: Dict[int, Dict[int, WinLossDrawCounts]] = defaultdict(lambda: defaultdict(WinLossDrawCounts))
        self.ratings: Dict[int, float] = {}  # mcts_gen -> rating

        self.db_filename = self.organizer.ratings_db_filename
        self._conn = None

    @staticmethod
    def get_mcts_player_name(gen: int):
        return f'MCTS-{gen}'

    def get_mcts_player_str(self, gen: int):
        name = Arena.get_mcts_player_name(gen)
        model = self.organizer.get_model_filename(gen)
        return f'--type=MCTS-C --name={name} -i {Params.mcts_iters} -m {model} -n {Params.num_search_threads}'

    @staticmethod
    def get_reference_player_name(strength: int):
        return f'ref-{strength}'

    def get_reference_player_str(self, strength: int):
        name = Arena.get_reference_player_name(strength)
        family = self.game_spec.reference_player_family
        type_str = family.type_str
        strength_param = family.strength_param
        return f'--type={type_str} --name={name} {strength_param}={strength}'

    def create_cmd(self, mcts_gen: int, ref_strength: int, n_games: int) -> str:
        ps1 = self.get_mcts_player_str(mcts_gen)
        ps2 = self.get_reference_player_str(ref_strength)
        binary = self.organizer.get_latest_binary()
        assert binary is not None
        cmd = f'{binary} -G {n_games} -p {Params.parallelism_factor} --player "{ps1}" --player "{ps2}"'
        return cmd

    def test_mcts_vs_ref(self, mcts_gen: int, ref_strength: int) -> WinLossDrawCounts:
        """
        Runs a match, updates database with result, and returns the update win/loss/draw counts.
        """
        assert self.min_ref_strength <= ref_strength <= self.max_ref_strength, (mcts_gen, ref_strength)
        counts = self.match_data[mcts_gen][ref_strength]
        n_games = Params.n_games - counts.n_games
        if n_games <= 0:
            return counts
        cmd = self.create_cmd(mcts_gen, ref_strength, n_games)
        mcts_name = Arena.get_mcts_player_name(mcts_gen)
        ref_name = Arena.get_reference_player_name(ref_strength)
        logger.info(f'[{self.tag}] Running {mcts_name} vs {ref_name} match: {cmd}')

        proc = subprocess_util.Popen(cmd)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f'proc exited with code {proc.returncode}')
        record = extract_match_record(stdout)
        counts += record.get(0)
        self.commit_counts(mcts_gen, ref_strength, counts)
        logger.info(f'[{self.tag}] Match result: {counts}')
        return counts

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_filename)
        return self._conn

    def init_db(self):
        if os.path.isfile(self.db_filename):
            if Params.clear_db:
                os.remove(self.db_filename)
            else:
                return

        logger.info(f'[{self.tag}] Initializing database')
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS matches (
            mcts_gen INT,
            mcts_iters INT,
            ref_strength INT,
            mcts_wins INT,
            draws INT,
            ref_wins INT);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (mcts_gen, mcts_iters, ref_strength);""")

        c.execute("""CREATE TABLE IF NOT EXISTS ratings (
            mcts_gen INT,
            mcts_iters INT,
            n_games INT,
            rating FLOAT);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (mcts_gen, mcts_iters);""")

        c.execute("""CREATE TABLE IF NOT EXISTS x_values (
            mcts_gen INT,
            n_games INT,
            runtime FLOAT,
            n_evaluated_positions BIGINT,
            n_batches_evaluated INT);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON x_values (mcts_gen);""")

        self.conn.commit()

    def dump_x_var_data(self):
        c = self.conn.cursor()

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

        training_db_conn = open_readonly_conn(self.organizer.self_play_db_filename)

        c2 = training_db_conn.cursor()
        c2.execute(query1, missing_mcts_gen_list)
        results1 = list(c2.fetchall())

        c2.execute(query2, missing_mcts_gen_list)
        results2 = list(c2.fetchall())
        training_db_conn.close()

        runtime_dict = defaultdict(int)
        for gen, start_timestamp, end_timestamp in results2:
            runtime_dict[gen] += end_timestamp - start_timestamp

        values = []
        for gen, n_evaluated_positions, n_batches_evaluated, n_games in results1:
            runtime = runtime_dict[gen]
            x_value_tuple = (gen, n_games, runtime, n_evaluated_positions, n_batches_evaluated)
            values.append(x_value_tuple)

        c.executemany('INSERT INTO x_values VALUES (?, ?, ?, ?, ?)', values)
        self.conn.commit()
        logger.info(f'[{self.tag}] Dumped {len(values)} rows of x var data...')

    def load_past_data(self):
        logger.info(f'[{self.tag}] Loading past data...')
        c = self.conn.cursor()
        res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE mcts_iters = ?',
                        (Params.mcts_iters,))
        for mcts_gen, ref_strength, mcts_wins, draws, ref_wins in res.fetchall():
            self.match_data[mcts_gen][ref_strength] = WinLossDrawCounts(mcts_wins, ref_wins, draws)

        res = c.execute('SELECT mcts_gen, rating FROM ratings WHERE mcts_iters = ? AND n_games >= ?',
                        (Params.mcts_iters, Params.n_games))
        for mcts_gen, rating in res.fetchall():
            self.ratings[mcts_gen] = rating

        count = 0
        for mcts_gen in sorted(self.match_data):
            rating = self.compute_rating(mcts_gen)
            if rating is not None:
                count += 1

        logger.info(f'[{self.tag}] Loaded {count} ratings')

    def compute_rating(self, mcts_gen: int):
        """
        Computes rating for a given MCTS generation based on available match data. Stores the rating in self.ratings
        and returns it.
        """
        if mcts_gen in self.ratings:
            return self.ratings[mcts_gen]

        ref_dict = {k: v for k, v in self.match_data[mcts_gen].items() if v.n_games >= Params.n_games}
        for strength in sorted(ref_dict):
            counts = ref_dict[strength]
            if counts.win_rate() < 0.5:
                # first level with losing record
                if strength == self.min_ref_strength:
                    self.ratings[mcts_gen] = 0.0
                    return 0.0

                prev_strength = strength - 1
                if prev_strength in ref_dict:
                    # rating = ((d-1) * (0.5 - w(d)) + d * (w(d-1) - 0.5)) / (w(d-1) - w(d))
                    prev_counts = ref_dict[prev_strength]
                    prev_weight = 0.5 - counts.win_rate()
                    weight = prev_counts.win_rate() - 0.5
                    rating = (prev_strength * prev_weight + strength * weight) / (weight + prev_weight)
                    self.ratings[mcts_gen] = rating
                    return rating

                # we don't have enough data to compute a rating
                break

        if self.max_ref_strength in ref_dict:
            self.ratings[mcts_gen] = self.max_ref_strength
            return self.max_ref_strength

        return None

    def get_next_work_item(self) -> Optional[WorkItem]:
        """
        Returns the next work item.

        Description of selection algorithm:

        Let G be the set of gens that we have graded thus far, and let M be the max generation that exists in the
        models directory.

        If M is at least 10 greater than the max element of G, then we return M. This is to keep up with a currently
        running alphazero run.

        Otherwise, if 1 is not in G, then we return 1.

        Finally, we find the largest gap in G, and return the midpoint of that gap. If G is fully saturated, we return
        M if M is not in G. If no such number exists, we return None.
        """
        latest_gen = self.organizer.get_latest_generation()
        if latest_gen == 0:
            return None

        graded_gens = list(sorted(self.ratings.keys()))
        if not graded_gens:
            return WorkItem(latest_gen, None, latest_gen, False, self)

        max_graded_gen = graded_gens[-1]
        max_graded_gen_rating = self.ratings.get(max_graded_gen, None)
        assert latest_gen >= max_graded_gen
        latest_gap = latest_gen - max_graded_gen
        if latest_gap >= 10:
            return WorkItem(latest_gen, max_graded_gen_rating, latest_gap, True, self)

        if 1 not in self.ratings:
            return WorkItem(1, 0.5, max_graded_gen, False, self)

        graded_gen_gaps = [graded_gens[i] - graded_gens[i - 1] for i in range(1, len(graded_gens))]
        gap_index_pairs = [(gap, i) for i, gap in enumerate(graded_gen_gaps)]
        if gap_index_pairs:
            largest_gap, largest_gap_index = max(gap_index_pairs)
            if largest_gap > 1:
                if largest_gap <= latest_gap:
                    return WorkItem(latest_gen, max_graded_gen_rating, latest_gap, True, self)
                prev_gen = graded_gens[largest_gap_index]
                next_gen = graded_gens[largest_gap_index + 1]
                gen = prev_gen + largest_gap // 2
                assert gen not in self.ratings, gen
                rating = 0.5 * (self.ratings[prev_gen] + self.ratings[next_gen])
                return WorkItem(gen, rating, largest_gap, False, self)

        if latest_gen != max_graded_gen:
            return WorkItem(latest_gen, self.ratings[max_graded_gen], latest_gen - max_graded_gen, True, self)

        return None

    def process(self, item: WorkItem):
        gen = item.mcts_gen
        est_rating = item.est_rating
        self.run_matches(gen, est_rating)
        rating = self.compute_rating(gen)
        assert rating is not None, gen
        self.commit_rating(gen, rating)

    def commit_rating(self, gen: int, rating: float):
        rating_tuple = (gen, Params.mcts_iters, Params.n_games, rating)
        c = self.conn.cursor()
        c.execute('REPLACE INTO ratings VALUES (?, ?, ?, ?)', rating_tuple)
        self.conn.commit()

    def commit_counts(self, mcts_gen: int, ref_strength: int, counts: WinLossDrawCounts):
        match_tuple = (mcts_gen, Params.mcts_iters, ref_strength, counts.win, counts.draw, counts.loss)
        c = self.conn.cursor()
        c.execute('REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?)', match_tuple)
        self.conn.commit()

    def run_matches(self, gen: int, est_rating: Optional[float]):
        """
        Let M(gen) be the MCTS agent using model gen, and for each strength s, let F[s] be the ref agent using strength
        s.

        Runs enough matches until we have identified a critical strength s* such that M(gen) is losing to F[s*] but not
        losing to F[s* - 1].

        The est_rating value is a hint as to where we should start looking for this critical strength.

        When this method is called, we may have some partial data for some strengths; the implementation intelligently
        utilizes the partial information along with the rating hint to try to find the critical strength at minimal
        cost.

        Throughout this method, we assume that win-rate is a non-decreasing function of ref-strength.
        """
        logger.info(f'[{self.tag}] Running matches for gen %s (est rating %s)' %
                    (gen, None if est_rating is None else '%.3f' % est_rating))

        ref_dict = {k: v for k, v in self.match_data[gen].items() if v.n_games >= Params.n_games}
        right_strengths = [strength for strength, counts in ref_dict.items() if counts.win_rate() <= 0.5]
        left_strengths = [strength for strength, counts in ref_dict.items() if counts.win_rate() > 0.5]

        if right_strengths:
            min_ref_winning_strength = min(right_strengths)
            left_strengths = [strength for strength in left_strengths if strength < min_ref_winning_strength]

        min_right_strength = min(right_strengths, default=self.max_ref_strength + 1)
        max_left_strength = max(left_strengths, default=-1)
        assert max_left_strength < min_right_strength
        if est_rating is None:
            est_rating = 0.5 * (min_right_strength + max_left_strength)
        else:
            if est_rating <= max_left_strength or est_rating >= min_right_strength:
                est_rating = None
        self.run_matches_helper(gen, est_rating, max_left_strength, min_right_strength)
        logger.info('[%s] Computed gen-%d rating: %.3f' % (self.tag, gen, self.compute_rating(gen)))

    def run_matches_helper(self, gen: int, est_rating: Optional[float],
                           max_left_strength: int, min_right_strength: int):
        """
        Helper method to run_matches().
        """
        est_rating_str = 'None' if est_rating is None else '%.3f' % est_rating
        logger.info('[%s] run_matches_helper(gen=%d, est_rating=%s, max_left_strength=%d, min_right_strength=%d)' %
                    (self.tag, gen, est_rating_str, max_left_strength, min_right_strength))
        assert max_left_strength < min_right_strength
        if max_left_strength + 1 == min_right_strength:
            return

        if est_rating is not None:
            assert max_left_strength < est_rating < min_right_strength
            strength = int(round(est_rating))
            counts = self.test_mcts_vs_ref(gen, strength)
            if counts.win_rate() < 0.5:  # mcts has losing record against ref
                strength2 = strength - 1
                if strength2 >= self.min_ref_strength:
                    counts2 = self.test_mcts_vs_ref(gen, strength2)
                    if counts2.win_rate() < 0.5:  # commence binary search
                        self.run_matches_helper(gen, None, max_left_strength, strength2)
            else:
                strength2 = strength + 1
                if strength2 <= self.max_ref_strength:
                    counts2 = self.test_mcts_vs_ref(gen, strength2)
                    if counts2.win_rate() >= 0.5:  # commence binary search
                        self.run_matches_helper(gen, None, strength2, min_right_strength)
            return

        # binary search for the critical strength
        mid_strength = (max_left_strength + min_right_strength) // 2
        assert max_left_strength < mid_strength < min_right_strength
        counts = self.test_mcts_vs_ref(gen, mid_strength)
        if counts.win_rate() < 0.5:  # mcts has losing record against ref
            self.run_matches_helper(gen, None, max_left_strength, mid_strength)
        else:
            self.run_matches_helper(gen, None, mid_strength, min_right_strength)

    def prepare(self):
        self.init_db()
        self.dump_x_var_data()
        self.load_past_data()


def main():
    load_args()
    configure_logger()

    arenas = []
    for tag in Params.tags:
        common_params = CommonParams(Params.alphazero_dir, Params.game, tag)
        arenas.append(Arena(common_params))

    for arena in arenas:
        arena.prepare()

    num_retries = 5  # because of sqlite3 race condition

    retry_count = num_retries
    while True:
        try:
            [arena.dump_x_var_data() for arena in arenas]
            queue = [arena.get_next_work_item() for arena in arenas]
            queue = [item for item in queue if item is not None]
            if not queue:
                if Params.daemon_mode:
                    time.sleep(5)
                    retry_count = num_retries
                    continue
                else:
                    return

            queue.sort(key=lambda item: (item.recency_boost, item.rating_gap))
            for item in reversed(queue):
                item.arena.process(item)
                break
            retry_count = num_retries
        except sqlite3.OperationalError as e:
            logger.info(f'Caught sqlite3.OperationalError: {e}')
            retry_count -= 1
            if retry_count > 0:
                logger.info(f'Retrying in 2 seconds...')
                time.sleep(2)
            else:
                raise


if __name__ == '__main__':
    main()
