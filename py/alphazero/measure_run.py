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
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import games
from alphazero.manager import AlphaZeroManager
from alphazero.ratings import extract_match_record, WinLossDrawCounts
from config import Config
from util import subprocess_util
from util.py_util import timed_print
from util.str_util import inject_args


class Args:
    alphazero_dir: str
    game: str
    tag: str
    clear_db: bool
    n_games: int
    mcts_iters: int
    parallelism_factor: int
    binary: str
    daemon_mode: bool

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.game = args.game
        Args.tag = args.tag
        Args.clear_db = bool(args.clear_db)
        Args.n_games = args.n_games
        Args.mcts_iters = args.mcts_iters
        Args.parallelism_factor = args.parallelism_factor
        Args.binary = args.binary
        Args.daemon_mode = bool(args.daemon_mode)
        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-g', '--game', help='game to play (e.g. "c4")')
    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    parser.add_argument('-C', '--clear-db', action='store_true', help='clear everything from database')
    parser.add_argument('-n', '--n-games', type=int, default=100,
                        help='number of games to play per matchup (default: %(default)s))')
    parser.add_argument('-i', '--mcts-iters', type=int, default=300,
                        help='number of MCTS iterations per move (default: %(default)s)')
    parser.add_argument('-p', '--parallelism-factor', type=int, default=100,
                        help='parallelism factor (default: %(default)s)')
    parser.add_argument('-b', '--binary', help='optional binary')
    parser.add_argument('-D', '--daemon-mode', action='store_true', help='daemon mode (run forever)')

    args = parser.parse_args()
    Args.load(args)


class Arena:
    def __init__(self):
        self.game_type = games.get_game_type(Args.game)
        self.base_dir = os.path.join(Args.alphazero_dir, Args.game, Args.tag)
        self.manager = AlphaZeroManager(self.game_type, self.base_dir)

        self.min_ref_strength = self.game_type.reference_player_family.min_strength
        self.max_ref_strength = self.game_type.reference_player_family.max_strength

        # mcts_gen -> ref_strength -> WLD
        self.match_data: Dict[int, Dict[int, WinLossDrawCounts]] = defaultdict(lambda: defaultdict(WinLossDrawCounts))
        self.ratings: Dict[int, float] = {}  # mcts_gen -> rating

        self.db_filename = os.path.join(self.base_dir, 'ratings.db')
        self._conn = None

    @staticmethod
    def get_mcts_player_name(gen: int):
        return f'MCTS-{gen}'

    def get_binary(self, gen: int):
        if Args.binary:
            return Args.binary
        cmd = self.manager.get_player_cmd(gen)
        assert cmd is not None, gen
        return cmd.split()[0]

    def get_mcts_player_str(self, gen: int):
        cmd = self.manager.get_player_cmd(gen)
        assert cmd is not None, gen
        player_str = cmd[cmd.find('"') + 1: cmd.rfind('"')]
        name = Arena.get_mcts_player_name(gen)
        kwargs = {
            '--name': name,
            '-i': Args.mcts_iters,
        }
        return inject_args(player_str, kwargs)

    @staticmethod
    def get_reference_player_name(strength: int):
        return f'ref-{strength}'

    def get_reference_player_str(self, strength: int):
        name = Arena.get_reference_player_name(strength)
        family = self.game_type.reference_player_family
        type_str = family.type_str
        strength_param = family.strength_param
        return f'--type={type_str} --name={name} {strength_param}={strength}'

    def create_cmd(self, mcts_gen: int, ref_strength: int, n_games: int) -> str:
        ps1 = self.get_mcts_player_str(mcts_gen)
        ps2 = self.get_reference_player_str(ref_strength)
        binary = self.get_binary(mcts_gen)
        cmd = f'{binary} -G {n_games} -p {Args.parallelism_factor} --player "{ps1}" --player "{ps2}"'
        return cmd

    def test_mcts_vs_ref(self, mcts_gen: int, ref_strength: int) -> WinLossDrawCounts:
        """
        Runs a match, updates database with result, and returns the update win/loss/draw counts.
        """
        assert self.min_ref_strength <= ref_strength <= self.max_ref_strength, (mcts_gen, ref_strength)
        counts = self.match_data[mcts_gen][ref_strength]
        n_games = Args.n_games - counts.n_games
        if n_games <= 0:
            return counts
        cmd = self.create_cmd(mcts_gen, ref_strength, n_games)
        mcts_name = Arena.get_mcts_player_name(mcts_gen)
        ref_name = Arena.get_reference_player_name(ref_strength)
        timed_print(f'Running {mcts_name} vs {ref_name} match: {cmd}')

        proc = subprocess_util.Popen(cmd)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f'proc exited with code {proc.returncode}')
        record = extract_match_record(stdout)
        counts += record.get(0)
        self.commit_counts(mcts_gen, ref_strength, counts)
        timed_print('Match result:', counts)
        return counts

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_filename)
        return self._conn

    def dump_metadata(self):
        metadata_filename = os.path.join(self.base_dir, 'metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump({
                'n_games': Args.n_games,
                'mcts_iters': Args.mcts_iters,
            }, f, indent=4)
        timed_print('Dumped metadata to', metadata_filename)

    def init_db(self):
        if os.path.isfile(self.db_filename):
            if Args.clear_db:
                os.remove(self.db_filename)
            else:
                return

        timed_print('Initializing database')
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
        self.conn.commit()

    def load_past_data(self):
        timed_print('Loading past data...')
        c = self.conn.cursor()
        res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE mcts_iters = ?',
                        (Args.mcts_iters,))
        for mcts_gen, ref_strength, mcts_wins, draws, ref_wins in res.fetchall():
            self.match_data[mcts_gen][ref_strength] = WinLossDrawCounts(mcts_wins, ref_wins, draws)

        res = c.execute('SELECT mcts_gen, rating FROM ratings WHERE mcts_iters = ? AND n_games >= ?',
                        (Args.mcts_iters, Args.n_games))
        for mcts_gen, rating in res.fetchall():
            self.ratings[mcts_gen] = rating

        for mcts_gen in sorted(self.match_data):
            rating = self.compute_rating(mcts_gen)
            if rating is not None:
                print(f'Loaded mcts-{mcts_gen} rating: {rating:.3f}')

        timed_print('Done loading past data!')

    def compute_rating(self, mcts_gen: int):
        """
        Computes rating for a given MCTS generation based on available match data. Stores the rating in self.ratings
        and returns it.
        """
        if mcts_gen in self.ratings:
            return self.ratings[mcts_gen]

        ref_dict = {k: v for k, v in self.match_data[mcts_gen].items() if v.n_games >= Args.n_games}
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

    def select_next_mcts_gen_to_rate(self) -> Tuple[Optional[int], Optional[float]]:
        """
        Returns the next gen to rate, along with an estimate of its rating.

        Description of selection algorithm:

        Let G be the set of gens that we have graded thus far, and let M be the max generation that exists in the
        models directory.

        If M is at least 10 greater than the max element of G, then we return M.

        Otherwise, if 1 is not in G, then we return 1.

        Finally, we find the largest gap in G, and return the midpoint of that gap. If G is fully saturated, we return
        M if M is not in G. If no such number exists, we return None.
        """
        latest_gen = self.manager.get_latest_generation()
        if latest_gen == 0:
            return None, None

        graded_gens = list(sorted(self.ratings.keys()))
        if not graded_gens:
            return latest_gen, None

        max_graded_gen = graded_gens[-1]
        assert latest_gen >= max_graded_gen
        if latest_gen - max_graded_gen >= 10:
            return latest_gen, self.ratings.get(max_graded_gen, None)

        if 1 not in self.ratings:
            return 1, 0.5

        graded_gen_gaps = [graded_gens[i] - graded_gens[i - 1] for i in range(1, len(graded_gens))]
        gap_index_pairs = [(gap, i) for i, gap in enumerate(graded_gen_gaps)]
        assert gap_index_pairs
        largest_gap, largest_gap_index = max(gap_index_pairs)
        if largest_gap > 1:
            prev_gen = graded_gens[largest_gap_index]
            next_gen = graded_gens[largest_gap_index + 1]
            gen = prev_gen + largest_gap // 2
            assert gen not in self.ratings, gen
            rating = 0.5 * (self.ratings[prev_gen] + self.ratings[next_gen])
            return gen, rating

        if latest_gen != max_graded_gen:
            return latest_gen, self.ratings[max_graded_gen]

        return None, None

    def run(self):
        while True:
            gen, est_rating = self.select_next_mcts_gen_to_rate()

            if gen is None:
                if Args.daemon_mode:
                    time.sleep(5)
                    continue
                else:
                    return

            self.run_matches(gen, est_rating)
            rating = self.compute_rating(gen)
            assert rating is not None, gen
            self.commit_rating(gen, rating)

    def commit_rating(self, gen: int, rating: float):
        rating_tuple = (gen, Args.mcts_iters, Args.n_games, rating)
        c = self.conn.cursor()
        c.execute('REPLACE INTO ratings VALUES (?, ?, ?, ?)', rating_tuple)
        self.conn.commit()

    def commit_counts(self, mcts_gen: int, ref_strength: int, counts: WinLossDrawCounts):
        match_tuple = (mcts_gen, Args.mcts_iters, ref_strength, counts.win, counts.draw, counts.loss)
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
        timed_print('Running matches for gen %s (est rating %s)' %
                    (gen, None if est_rating is None else '%.3f' % est_rating))

        ref_dict = {k: v for k, v in self.match_data[gen].items() if v.n_games >= Args.n_games}
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
        timed_print('Computed gen-%d rating: %.3f' % (gen, self.compute_rating(gen)))

    def run_matches_helper(self, gen: int, est_rating: Optional[float],
                           max_left_strength: int, min_right_strength: int):
        """
        Helper method to run_matches().
        """
        timed_print('run_matches_helper(gen=%d, est_rating=%s, max_left_strength=%d, min_right_strength=%d)' %
                    (gen, 'None' if est_rating is None else '%.3f' % est_rating, max_left_strength, min_right_strength))
        assert max_left_strength < min_right_strength
        if max_left_strength + 1 == min_right_strength:
            return

        if est_rating is not None:
            assert max_left_strength < est_rating < min_right_strength
            strength = int(round(est_rating))
            counts = self.test_mcts_vs_ref(gen, strength)
            if counts.win_rate() < 0.5:  # mcts has losing record against ref
                strength2 = strength - 1
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

    def launch(self):
        self.dump_metadata()
        self.init_db()
        self.load_past_data()
        self.run()


def main():
    load_args()
    arena = Arena()
    arena.launch()


if __name__ == '__main__':
    main()