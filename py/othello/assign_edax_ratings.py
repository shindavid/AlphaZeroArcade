#!/usr/bin/env python3

"""
Edax agents can be configured to use a fixed search depth.

Informally, if a player has equal skill level to an edax agent of depth d, we can say that the player has an
"edax rating of d".

This script takes an alphazero othello run directory and estimates the edax rating curve for the generations of
agents in that run.

More exact definition of an edax rating:

For positive d, let E(d) be an edax agent of depth d. Let E(0) denote the random agent.

For a player P, let d be the smallest integer such that P has a losing record against E(d). This implies that P has a
non-losing record against E(d-1). Let w(d) be the win-rate of P against E(d), and let w(d-1) be the win-rate of P
against E(d-1). We must have w(d) < 0.5 <= w(d-1). Then we define the edax rating of P to be the weighted-average of
(d-1) and d, where the weights are proportional to the distances of w(d) and w(d-1) from 0.5.

More precisely, the edax rating of P is:

((d-1) * (0.5 - w(d)) + d * (w(d-1) - 0.5)) / (w(d-1) - w(d))

Some illustrative examples, using a plausible evolution of win rates as P progresses from E(7)-level to E(8)-level:

* If P wins 50% against E(7) and 0% against E(8), then the weights for 7 and 8 are 0.5-0=0.5 and 0.5-0.5=0,
respectively, resulting in a rating of 7*0.5 / 0.5 = 7

* If P wins 60% against E(7) and 0% against E(8), then the weights for 7 and 8 are 0.5-0=0.5 and 0.6-0.5=0.1,
respectively, resulting in a rating of (7*0.5 + 8*0.1) / (0.5 + 0.1) = 7.166...

* If P wins 70% against E(7) and 20% against E(8), then the weights for 7 and 8 are 0.5-0.2=0.3 and 0.7-0.5=0.2,
respectively, resulting in a rating of (7*0.3 + 8*0.2) / (0.3 + 0.2) = 7.4

* If P wins 90% against E(7) and 40% against E(8), then the weights for 7 and 8 are 0.5-0.4=0.1 and 0.9-0.5=0.4,
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
    tag: str
    clear_db: bool
    n_games: int
    mcts_iters: int
    max_depth: int
    parallelism_factor: int
    binary: str
    daemon_mode: bool

    @staticmethod
    def load(args):
        Args.alphazero_dir = args.alphazero_dir
        Args.tag = args.tag
        Args.clear_db = bool(args.clear_db)
        Args.n_games = args.n_games
        Args.mcts_iters = args.mcts_iters
        Args.max_depth = args.max_depth
        Args.parallelism_factor = args.parallelism_factor
        Args.binary = args.binary
        Args.daemon_mode = bool(args.daemon_mode)
        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    cfg.add_parser_argument('alphazero_dir', parser, '-d', '--alphazero-dir', help='alphazero directory')
    parser.add_argument('-C', '--clear-db', action='store_true', help='clear everything from database')
    parser.add_argument('-n', '--n-games', type=int, default=100,
                        help='number of games to play per matchup (default: %(default)s))')
    parser.add_argument('-i', '--mcts-iters', type=int, default=300,
                        help='number of MCTS iterations per move (default: %(default)s)')
    parser.add_argument('-m', '--max-depth', type=int, default=21,
                        help='max edax depth (default: %(default)s)')
    parser.add_argument('-p', '--parallelism-factor', type=int, default=100,
                        help='parallelism factor (default: %(default)s)')
    parser.add_argument('-b', '--binary', help='optional binary')
    parser.add_argument('-D', '--daemon-mode', action='store_true', help='daemon mode (run forever)')

    args = parser.parse_args()
    Args.load(args)


class Arena:
    def __init__(self):
        othello = games.get_game_type('othello')
        self.base_dir = os.path.join(Args.alphazero_dir, 'othello', Args.tag)
        self.manager = AlphaZeroManager(othello, self.base_dir)

        # mcts_gen -> edax_depth -> WLD
        self.match_data: Dict[int, Dict[int, WinLossDrawCounts]] = defaultdict(lambda: defaultdict(WinLossDrawCounts))
        self.ratings: Dict[int, float] = {}  # mcts_gen -> rating

        self.db_filename = os.path.join(self.base_dir, 'edax.db')
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
    def get_edax_player_name(depth: int):
        if depth == 0:
            return 'Random'
        return f'edax-{depth}'

    @staticmethod
    def get_edax_player_str(depth: int):
        name = Arena.get_edax_player_name(depth)
        if depth == 0:
            return f'--type=Random --name={name}'
        return f'--type=edax --name={name} --depth={depth}'

    def create_cmd(self, mcts_gen: int, edax_depth: int, n_games: int) -> str:
        ps1 = self.get_mcts_player_str(mcts_gen)
        ps2 = Arena.get_edax_player_str(edax_depth)
        binary = self.get_binary(mcts_gen)
        cmd = f'{binary} -G {n_games} -p {Args.parallelism_factor} --player "{ps1}" --player "{ps2}"'
        return cmd

    def test_mcts_vs_edax(self, mcts_gen: int, edax_depth: int) -> WinLossDrawCounts:
        """
        Runs a match, updates database with result, and returns the update win/loss/draw counts.
        """
        assert 0 <= edax_depth <= Args.max_depth, (mcts_gen, edax_depth, Args.max_depth)
        counts = self.match_data[mcts_gen][edax_depth]
        n_games = Args.n_games - counts.n_games
        if n_games <= 0:
            return counts
        cmd = self.create_cmd(mcts_gen, edax_depth, n_games)
        timed_print(f'Running mcts-{mcts_gen} vs edax-{edax_depth} match: {cmd}')

        proc = subprocess_util.Popen(cmd)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f'proc exited with code {proc.returncode}')
        record = extract_match_record(stdout)
        counts += record.get(0)
        self.commit_counts(mcts_gen, edax_depth, counts)
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
            edax_depth INT,
            mcts_wins INT,
            draws INT,
            edax_wins INT);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (mcts_gen, mcts_iters, edax_depth);""")

        c.execute("""CREATE TABLE IF NOT EXISTS ratings (
            mcts_gen INT,
            mcts_iters INT,
            n_games INT,
            edax_rating FLOAT);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (mcts_gen, mcts_iters);""")
        self.conn.commit()

    def load_past_data(self):
        timed_print('Loading past data...')
        c = self.conn.cursor()
        res = c.execute('SELECT mcts_gen, edax_depth, mcts_wins, draws, edax_wins FROM matches WHERE mcts_iters = ?',
                        (Args.mcts_iters,))
        for mcts_gen, edax_depth, mcts_wins, draws, edax_wins in res.fetchall():
            self.match_data[mcts_gen][edax_depth] = WinLossDrawCounts(mcts_wins, edax_wins, draws)

        res = c.execute('SELECT mcts_gen, edax_rating FROM ratings WHERE mcts_iters = ? AND n_games >= ?',
                        (Args.mcts_iters, Args.n_games))
        for mcts_gen, edax_rating in res.fetchall():
            self.ratings[mcts_gen] = edax_rating

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

        edax_dict = {k: v for k, v in self.match_data[mcts_gen].items() if v.n_games >= Args.n_games}
        for level in sorted(edax_dict):
            counts = edax_dict[level]
            if counts.win_rate() < 0.5:
                # first level with losing record
                if level == 0:
                    self.ratings[mcts_gen] = 0.0
                    return 0.0

                prev_level = level - 1
                if prev_level in edax_dict:
                    # rating = ((d-1) * (0.5 - w(d)) + d * (w(d-1) - 0.5)) / (w(d-1) - w(d))
                    prev_counts = edax_dict[prev_level]
                    prev_weight = 0.5 - counts.win_rate()
                    weight = prev_counts.win_rate() - 0.5
                    rating = (prev_level * prev_weight + level * weight) / (weight + prev_weight)
                    self.ratings[mcts_gen] = rating
                    return rating

                # we don't have enough data to compute a rating
                break

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

    def commit_counts(self, mcts_gen: int, edax_depth: int, counts: WinLossDrawCounts):
        # res = c.execute('SELECT mcts_gen, edax_depth, mcts_wins, draws, edax_wins FROM matches')

        match_tuple = (mcts_gen, Args.mcts_iters, edax_depth, counts.win, counts.draw, counts.loss)
        c = self.conn.cursor()
        c.execute('REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?)', match_tuple)
        self.conn.commit()

    def run_matches(self, gen: int, est_rating: Optional[float]):
        """
        Let M(gen) be the MCTS agent using model gen, and for each depth d, let E(d) be the edax agent using depth d.

        Runs enough matches until we have identified a critical depth d* such that M(gen) is losing to E(d*) but not
        losing to E(d* - 1).

        The est_rating value is a hint as to where we should start looking for this critical depth.

        When this method is called, we may have some partial data for some depths; the implementation intelligently
        utilizes the partial information along with the rating hint to try to find the critical depth at minimal cost.

        Throughout this method, we assume that win-rate is a non-decreasing function of edax-depth.
        """
        timed_print('Running matches for gen %s (est rating %s)' %
                    (gen, None if est_rating is None else '%.3f' % est_rating))

        edax_dict = {k: v for k, v in self.match_data[gen].items() if v.n_games >= Args.n_games}
        right_depths = [depth for depth, counts in edax_dict.items() if counts.win_rate() <= 0.5]
        left_depths = [depth for depth, counts in edax_dict.items() if counts.win_rate() > 0.5]

        if right_depths:
            min_edax_winning_depth = min(right_depths)
            left_depths = [depth for depth in left_depths if depth < min_edax_winning_depth]

        min_right_depth = min(right_depths, default=Args.max_depth + 1)
        max_left_depth = max(left_depths, default=-1)
        assert max_left_depth < min_right_depth
        if est_rating is None:
            est_rating = 0.5 * (min_right_depth + max_left_depth)
        else:
            if est_rating <= max_left_depth or est_rating >= min_right_depth:
                est_rating = None
        self.run_matches_helper(gen, est_rating, max_left_depth, min_right_depth)
        timed_print('Computed gen-%d rating: %.3f' % (gen, self.compute_rating(gen)))

    def run_matches_helper(self, gen: int, est_rating: Optional[float], max_left_depth: int, min_right_depth: int):
        """
        Helper method to run_matches().
        """
        timed_print('run_matches_helper(gen=%d, est_rating=%s, max_left_depth=%d, min_right_depth=%d)' %
                    (gen, 'None' if est_rating is None else '%.3f' % est_rating, max_left_depth, min_right_depth))
        assert max_left_depth < min_right_depth
        if max_left_depth + 1 == min_right_depth:
            return

        if est_rating is not None:
            assert max_left_depth < est_rating < min_right_depth
            depth = int(round(est_rating))
            counts = self.test_mcts_vs_edax(gen, depth)
            if counts.win_rate() < 0.5:  # mcts has losing record against edax
                depth2 = depth - 1
                counts2 = self.test_mcts_vs_edax(gen, depth2)
                if counts2.win_rate() < 0.5:  # commence binary search
                    self.run_matches_helper(gen, None, max_left_depth, depth2)
            else:
                depth2 = depth + 1
                counts2 = self.test_mcts_vs_edax(gen, depth2)
                if counts2.win_rate() >= 0.5:  # commence binary search
                    self.run_matches_helper(gen, None, depth2, min_right_depth)
            return

        # binary search for the critical depth
        mid_depth = (max_left_depth + min_right_depth) // 2
        assert max_left_depth < mid_depth < min_right_depth
        counts = self.test_mcts_vs_edax(gen, mid_depth)
        if counts.win_rate() < 0.5:  # mcts has losing record against edax
            self.run_matches_helper(gen, None, max_left_depth, mid_depth)
        else:
            self.run_matches_helper(gen, None, mid_depth, min_right_depth)

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
