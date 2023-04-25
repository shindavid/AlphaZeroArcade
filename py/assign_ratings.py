#!/usr/bin/env python3

"""
Assigns skill ratings to a fixed set of agents produced by the alphazero main loop.

We use the basic Bradley-Terry model, without the generalized capabilities of first-player advantage or draws. This
model models agent i's skill as a parameter beta_i, and predicts the expected win-rate in a match between agent i
and agent j as:

P(i beats j) = e^beta_i / (e^beta_i + e^beta_j)

To estimate the beta_i's, we run a series of matches between the agents. Below is a description of the methodology.

Let N be the total number of agents in our system. As a baseline, we can use the agents produced by the alphazero main
loop, with each agent configured to use i=256 MCTS iterations. If we want, we can produce additional agents by
parameterizing those baseline agents differently (e.g., using a different number of MCTS iterations). We can also
add additional non-MCTS agents (Random, Perfect for c4, etc.).

We have some expectations on the relative skill level of these agents. For example, we expect the generation-k agent
to be inferior to the generation-(k+1) agent. If we choose to use different parameterizations of the same generation,
we have expectations amongst those. For example, all else equal, we expect that increasing the number of MCTS
iterations increases the skill level of the agent. We also expect that the random agent will be inferior to all other
agents, and that the perfect agent will be superior to all other agents. We can encode all these expectations into an
(n, n)-shaped boolean "Expected Relative Skill Matrix", E. The (i, j)-th entry of this matrix will be true if we
expect agent i to be inferior to agent j.

Let G be the N-node graph with directed edges defined by this matrix E. G should not contain cycles if the skill
expectations are coherent. Thus, G is a DAG, and we can extend it to its transitive closure by adding edges between
nodes u and v if there is a directed path from u to v in G.

We initialize our game records data by first including a fractional virtual win for agent i over agent j for each edge
(i, j) in G. This softly encodes our expectations on the relative skill levels of the agents.

After this, we repeatedly sample an edge of G and then run a set of matches between the corresponding agents. Consider
the digraph H formed by drawing an edge from u to v whenever u records at least one win (or draw) against v. We
require H to be strongly connected before we can make proper predictions, so our match sampling will initially favor
sampling edges that can potentially connect clusters of H. The sampling details can be found in the code.
"""
import argparse
import collections
import os
import random
import sqlite3
import time
from typing import Optional, Dict, List

from natsort import natsorted
import numpy as np

from config import Config
from util import subprocess_util
from util.graph_util import transitive_closure, direct_children
from util.py_util import timed_print


BETA_SCALE_FACTOR = 100.0 / np.log(1/.36 - 1)  # 100-point difference corresponds to 64% win-rate to match Elo


class Args:
    c4_base_dir_root: str
    tag: str
    clear_db: bool
    n_games: int
    mcts_iters: int
    parallelism_factor: int
    num_rounds: int

    @staticmethod
    def load(args):
        Args.c4_base_dir_root = args.c4_base_dir_root
        Args.tag = args.tag
        Args.clear_db = bool(args.clear_db)
        Args.n_games = args.n_games
        Args.mcts_iters = args.mcts_iters
        Args.parallelism_factor = args.parallelism_factor
        Args.num_rounds = args.num_rounds
        assert Args.tag, 'Required option: -t'


def load_args():
    parser = argparse.ArgumentParser()
    cfg = Config.instance()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')
    cfg.add_parser_argument('c4.base_dir_root', parser, '-d', '--c4-base-dir-root',
                            help='base-dir-root for game/model files')
    parser.add_argument('-C', '--clear-db', action='store_true', help='clear everything from database')
    parser.add_argument('-n', '--n-games', type=int, default=64,
                        help='number of games to play per matchup (default: %(default)s))')
    parser.add_argument('-i', '--mcts-iters', type=int, default=256,
                        help='number of MCTS iterations per move (default: %(default)s)')
    parser.add_argument('-p', '--parallelism-factor', type=int, default=64,
                        help='parallelism factor (default: %(default)s)')
    parser.add_argument('-r', '--num-rounds', type=int, default=10,
                        help='num round-robin rounds (default: %(default)s)')

    args = parser.parse_args()
    Args.load(args)


def inject_arg(cmdline: str, arg_name: str, arg_value: str):
    """
    Takes a cmdline and adds {arg_name}={arg_value} into it, overriding any existing value for {arg_name}.
    """
    assert arg_name.startswith('-'), arg_name
    tokens = cmdline.split()
    for i, token in enumerate(tokens):
        if token == arg_name:
            tokens[i+1] = arg_value
            return ' '.join(tokens)

        if token.startswith(f'{arg_name}='):
            tokens[i] = f'{arg_name}={arg_value}'
            return ' '.join(tokens)

    return f'{cmdline} {arg_name} {arg_value}'


def inject_args(cmdline: str, kwargs: dict):
    for arg_name, arg_value in kwargs.items():
        cmdline = inject_arg(cmdline, arg_name, arg_value)
    return cmdline


def int_parse(s: str, prefix: str):
    assert s.startswith(prefix), s
    return int(s[len(prefix):])


class WinLossDrawCounts:
    def __init__(self, win=0, loss=0, draw=0):
        self.win = win
        self.loss = loss
        self.draw = draw

    def __iadd__(self, other):
        self.win += other.win
        self.loss += other.loss
        self.draw += other.draw
        return self

    def __str__(self):
        return f'W{self.win} L{self.loss} D{self.draw}'


class MatchRecord:
    def __init__(self):
        self.counts: Dict[int, WinLossDrawCounts] = collections.defaultdict(WinLossDrawCounts)

    def update(self, player_id: int, counts: WinLossDrawCounts):
        self.counts[player_id] += counts

    def get(self, player_id: int) -> WinLossDrawCounts:
        return self.counts[player_id]

    def empty(self) -> bool:
        return len(self.counts) == 0


def extract_match_record(stdout: str) -> MatchRecord:
    """
    ...
    All games complete!
    P0 W40 L24 D0 [40]
    P1 W24 L40 D0 [24]
    ...
    """
    record = MatchRecord()
    for line in stdout.splitlines():
        tokens = line.split()
        if len(tokens) > 1 and tokens[0][0] == 'P' and tokens[0][1:].isdigit():
            player_id = int_parse(tokens[0], 'P')
            win = int_parse(tokens[1], 'W')
            loss = int_parse(tokens[2], 'L')
            draw = int_parse(tokens[3], 'D')
            counts = WinLossDrawCounts(win, loss, draw)
            record.update(player_id, counts)

    assert not record.empty(), stdout
    return record


def construct_cmd(binary: str,
                  player_str: str,
                  binary_kwargs: Optional[dict] = None,
                  player_kwargs: Optional[dict] = None)-> str:
    cmd = binary
    if binary_kwargs:
        for key, value in binary_kwargs.items():
            assert key.startswith('-'), key
            cmd += f' {key} {value}'

    if player_kwargs:
        for key, value in player_kwargs.items():
            player_str = inject_arg(player_str, key, str(value))

    cmd += f' --player "{player_str}"'
    return cmd


class Agent:
    def __init__(self, cmd: str, *,
                 index: int = -1,
                 rand: bool = False,
                 perfect: bool = False,
                 gen: Optional[int] = None,
                 iters: Optional[int] = None,
                 row_id: Optional[int] = None):
        """
        cmd looks like:

        <binary> --player "--type=MCTS-C --nnet-filename /media/dshin/c4f/models/gen-10.ptj"

        See documentation at top of file for description of "special".
        """
        assert cmd.count('"') == 2, cmd
        tokens = cmd.split()
        assert tokens[1] == '--player', cmd
        assert '--player' not in tokens[3:], cmd
        assert tokens[2].startswith('"'), cmd
        assert tokens[-1].endswith('"'), cmd

        self.cmd = cmd
        self.binary = tokens[0]
        self.player_str = cmd[cmd.find('"') + 1: cmd.rfind('"')]

        self.index = index
        self.rand = rand
        self.perfect = perfect
        self.gen = gen
        self.iters = iters
        self.row_id = row_id

    def get_cmd(self, binary_kwargs: Optional[dict] = None, player_kwargs: Optional[dict] = None):
        return construct_cmd(self.binary, self.player_str, binary_kwargs, player_kwargs)

    def __str__(self):
        return f'Agent("{self.player_str}")'

    def __repr__(self):
        return str(self)


class Arena:
    def __init__(self):
        self.c4_base_dir = os.path.join(Args.c4_base_dir_root, Args.tag)
        assert os.path.isdir(self.c4_base_dir), self.c4_base_dir
        self.arena_dir = os.path.join(self.c4_base_dir, 'arena')
        os.makedirs(self.arena_dir, exist_ok=True)

        self.agents: List[Agent] = []
        self.agent_dict: Dict[int, Agent] = {}
        self.E = None  # expected relative skill matrix
        self.beta = None  # skill estimates
        self.virtual_wins = None
        self.real_wins = None
        self.db_filename = os.path.join(self.arena_dir, 'arena.db')
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_filename)
        return self._conn

    def init_db(self):
        if os.path.isfile(self.db_filename):
            if Args.clear_db:
                os.remove(self.db_filename)
            else:
                return

        timed_print('Initializing database')
        c = self.conn.cursor()
        agents_table_exists = c.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="agents"').fetchone()
        if not agents_table_exists:
            c.execute("""CREATE TABLE agents (
                cmd,
                rand,
                perfect,
                gen,
                iters);
            """)
            c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON agents (cmd);""")

            players_dir = os.path.join(self.c4_base_dir, 'players')
            for filename in natsorted(os.listdir(players_dir)):  # gen-13.txt
                if filename.startswith('.') or not filename.endswith('.txt'):
                    continue

                gen = int(filename.split('.')[0].split('-')[1])
                if gen % 10 != 0:
                    # only use every 10th generation for now
                    continue

                with open(os.path.join(players_dir, filename)) as f:
                    cmd = f.read().strip()

                agent = Agent(cmd)
                cmd_with_iters = agent.get_cmd(player_kwargs={'-i': Args.mcts_iters})
                cmd_tuples = [(cmd_with_iters, False, False, gen, Args.mcts_iters)]
                if gen == 0:
                    rand_cmd = f'{agent.binary} --player "--type=Random"'
                    perfect_cmd = f'{agent.binary} --player "--type=Perfect"'
                    cmd4 = agent.get_cmd(player_kwargs={'-i': 4})
                    cmd16 = agent.get_cmd(player_kwargs={'-i': 16})
                    cmd64 = agent.get_cmd(player_kwargs={'-i': 64})

                    cmd_tuples = [
                        (rand_cmd, True, False, -1, 0),
                        (perfect_cmd, False, True, -1, 0),
                        (cmd4, False, False, gen, 4),
                        (cmd16, False, False, gen, 16),
                        (cmd64, False, False, gen, 64),
                    ] + cmd_tuples

                c.executemany('INSERT INTO agents VALUES (?, ?, ?, ?, ?)', cmd_tuples)

        # TODO: generalize below table for multiplayer games
        c.execute("""CREATE TABLE IF NOT EXISTS matches (
            agent_id1 INT, 
            agent_id2 INT, 
            wins1 INT, 
            draws INT, 
            wins2 INT);
        """)
        c.execute("""CREATE INDEX IF NOT EXISTS matches_agent_id1 ON matches (agent_id1);""")
        c.execute("""CREATE INDEX IF NOT EXISTS matches_agent_id2 ON matches (agent_id2);""")

        c.execute("""CREATE TABLE IF NOT EXISTS ratings (
            agent_id INT,
            match_id INT,
            beta REAL);
        """)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (agent_id, match_id);""")
        self.conn.commit()

    def init_agents(self):
        c = self.conn.cursor()
        res = c.execute('SELECT rowid, cmd, rand, perfect, gen, iters FROM agents')
        for row_id, cmd, rand, perfect, gen, iters in res.fetchall():
            agent = Agent(cmd, index=len(self.agents), row_id=row_id, rand=rand, perfect=perfect, gen=gen, iters=iters)
            self.agents.append(agent)
            self.agent_dict[row_id] = agent

        n = len(self.agents)
        self.beta = np.zeros(n, dtype=np.float64)

        assert self.agents[0].rand
        assert all(not agent.rand for agent in self.agents[1:])

        timed_print(f'Loaded {n} agents')

    def init_expected_relative_skill_matrix(self):
        n = len(self.agents)
        self.E = np.zeros((n, n), dtype=bool)
        for i in range(n):
            agent_i = self.agents[i]
            if agent_i.rand:
                self.E[i] = 1
                continue
            if agent_i.perfect:
                self.E[:, i] = 1
                continue
            for j in range(n):
                if i == j:
                    continue
                agent_j = self.agents[j]
                if agent_j.rand or agent_j.perfect:
                    continue

                if agent_i.gen < agent_j.gen:
                    self.E[i, j] = 1
                    continue
                elif agent_i.gen == agent_j.gen and agent_i.iters < agent_j.iters:
                    self.E[i, j] = 1
                    continue

        np.fill_diagonal(self.E, 0)

        # Above construction should yield a matrix that is its own transitive closure
        # self.E = transitive_closure(self.E)
        timed_print(f'Constructed ({n}, {n})-shaped Expected Relative Skill Matrix')

    def add_virtual_wins(self):
        # If agent i is expected to be inferior to agent j, record .01 wins of i over j, and .1 wins for j over i.
        Ef = self.E.astype(float)
        self.virtual_wins = 0.1 * Ef.T + 0.01 * Ef
        timed_print(f'Added virtual wins')

    def load_matches(self):
        n = len(self.agents)
        self.real_wins = np.zeros((n, n), dtype=float)
        c = self.conn.cursor()
        res = c.execute('SELECT rowid, agent_id1, agent_id2, wins1, draws, wins2 FROM matches')
        match_count = 0
        for row_id, agent_id1, agent_id2, wins1, draws, wins2 in res.fetchall():
            match_count += 1
            agent1 = self.agent_dict[agent_id1]
            agent2 = self.agent_dict[agent_id2]

            i1 = agent1.row_id
            i2 = agent2.row_id
            self.real_wins[i1, i2] = wins1 + 0.5 * draws
            self.real_wins[i2, i1] = wins2 + 0.5 * draws
        timed_print(f'Loaded {match_count} matches')

    def launch(self):
        self.init_db()
        self.init_agents()
        self.init_expected_relative_skill_matrix()
        self.add_virtual_wins()
        self.load_matches()
        self.update_ratings()

        for r in range(Args.num_rounds):
            self.play_round(r)

    def play_round(self, round_num: int):
        """
        For now, we do something really simple. We randomly pick an agent that has not yet played a match this round,
        and pit it against a random opponent that it has not yet played over all rounds. Repeat until every agent has
        played at least one match this round.

        Later, we can make this more sophisticated, choosing matchups that are more likely to be informative.
        """
        timed_print(f'Round {round_num}')
        n = len(self.agents)
        played_this_round = np.zeros(n, dtype=bool)
        while not np.all(played_this_round):
            i = random.choice(np.where(~played_this_round)[0])
            played_this_round[i] = True

            candidates = np.where((self.real_wins[i] == 0) & (self.real_wins[:, i] == 0))[0]
            candidates[i] = False
            if not np.any(candidates):
                continue

            j = random.choice(candidates)
            played_this_round[j] = True

            self.play_match(self.agents[i], self.agents[j])
            self.update_ratings()
        self.commit_ratings()

    def play_match(self, agent1: Agent, agent2: Agent):
        timed_print('Playing match')
        timed_print(f'Agent 1: {agent1}')
        timed_print(f'Agent 2: {agent2}')

        binary_kwargs = {'-G': Args.n_games}

        player1_kwargs = {'--name': 'Player1'}
        player2_kwargs = {'--name': 'Player2'}

        port = 12345
        binary_kwargs['--port'] = port
        cmd1 = agent1.get_cmd(binary_kwargs=binary_kwargs, player_kwargs=player1_kwargs)
        cmd2 = agent2.get_cmd(binary_kwargs={'--remote-port': port}, player_kwargs=player2_kwargs)
        timed_print(f'cmd1: {cmd1}')
        proc1 = subprocess_util.Popen(cmd1)
        time.sleep(0.1)  # won't be necessary soon
        timed_print(f'cmd2: {cmd2}')
        subprocess_util.Popen(cmd2)

        stdout, stderr = proc1.communicate()
        if proc1.returncode:
            raise RuntimeError(f'proc1 exited with code {proc1.returncode}')
        record = extract_match_record(stdout)

        counts1 = record.get(0)
        counts2 = record.get(1)
        assert (counts1.win, counts1.loss, counts1.draw) == (counts2.loss, counts2.win, counts2.draw)
        timed_print(f'Agent 1: {counts1}')

        i = agent1.index
        j = agent2.index
        self.real_wins[i, j] += counts1.win + 0.5 * counts1.draw
        self.real_wins[j, i] += counts2.win + 0.5 * counts2.draw

        match_tuple = (agent1.row_id, agent2.row_id, counts1.win, counts1.draw, counts2.win)
        c = self.conn.cursor()
        c.execute('INSERT INTO matches VALUES (?, ?, ?, ?, ?)', match_tuple)
        self.conn.commit()

    def update_ratings(self):
        eps = 1e-6
        w = self.real_wins + self.virtual_wins
        assert np.all(w >= 0)
        assert w.diagonal().sum() == 0
        ww = w + w.T
        W = np.sum(w, axis=1)

        p = np.exp(self.beta / BETA_SCALE_FACTOR)
        k = 0
        while True:
            pp = p.reshape((-1, 1)) + p.reshape((1, -1))
            wp_sum = np.sum(ww / pp, axis=1)
            gradient = W / p - wp_sum
            max_gradient = np.max(np.abs(gradient))
            if max_gradient < eps:
                break

            q = W / wp_sum
            q /= q[0]  # so that Random agent's rating is 0
            p = q
            k += 1

        prev_beta = self.beta
        self.beta = np.log(p) * BETA_SCALE_FACTOR
        beta_delta = self.beta - prev_beta
        beta_delta_indices = list(sorted(range(len(self.beta)), key=lambda i: -np.abs(beta_delta[i])))

        timed_print('Updated ratings after %d iterations (top-3 beta changes: %.3f, %.3f, %.3f, max-beta:%.3f)' %
                    (k, beta_delta[beta_delta_indices[0]],
                     beta_delta[beta_delta_indices[1]], beta_delta[beta_delta_indices[2]], np.max(self.beta)))

    def commit_ratings(self):
        c = self.conn.cursor()

        match_id = c.execute('SELECT MAX(rowid) FROM matches').fetchone()[0]
        insert_tuples = []
        for agent in self.agents:
            agent_id = agent.row_id
            beta = self.beta[agent.index]
            insert_tuples.append((match_id, agent_id, beta))

        c.executemany('INSERT INTO ratings VALUES (?, ?, ?)', insert_tuples)
        self.conn.commit()
        timed_print(f'Committed ratings')


def main():
    load_args()
    arena = Arena()
    arena.launch()


if __name__ == '__main__':
    main()
