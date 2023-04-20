#!/usr/bin/env python3

"""
Assigns TrueSkill ratings to a fixed set of agents produced by the alphazero main loop. Each is configured to use
i=256 MCTS iterations.

The ratings are computed by running a series of matches between the agents. For now, the methodology looks like this:

for K in [0, 1, 2, ...]:
  - For each of the N agents, we uniformly randomly select an opponent, and pit the two against each other for G games.
  - Compute ratings for the N agents based on the cumulative N*(K+1)*G games so far, and export those ratings as R(K).

The plan is to study the progression of R(0), R(1), ..., to see if/how the ratings converge. We can also sample from
the N agents beforehand, to see if working with a smaller subset of agents leads to similar ratings.

Some additional agents are added to the set of agents to be rated:

- Random: an agent that chooses uniformly randomly from all legal moves.
- (c4-only) Perfect: An agent that plays according to the perfect strategy for the game.
- Gen-0 i=4: The zero-eth generation agent, but with 4 MCTS iterations.
- Gen-0 i=16: The zero-eth generation agent, but with 16 MCTS iterations.
- Gen-0 i=64: The zero-eth generation agent, but with 64 MCTS iterations.

The ratings will be calibrated so that Random has a rating of 0.

Additionally, we forcibly add round-robin matches between {Random, Gen-0 i=4, Gen-0 i=16, Gen-0 i=64} before starting
the loop.

Rationale: it is good to have a well-defined player like Random in the population with a fixed rating, as it helps with
the interpretability of the ratings. However, in order to properly calibrate Random's rating relative to the rest of the
population, we need to include matchups where Random can win. Against the default setting of i=256 MCTS iterations,
Random empirically loses 100% or near-100% against even the zero-eth generation net. Empirically, however, Random is
able to win a non-negligible fraction of the time against the first generation net with 4, 16, or 64 MCTS iterations.
Furthermore, the Gen-0 i=64 agent is able to win a non-negligible fraction of the time against many i=256 agents, even
against opponents are advanced as Gen-50.
"""
import argparse
import os
import random
import sqlite3
from typing import Optional

import trueskill
from natsort import natsorted

from config import Config
from util.py_util import timed_print

"""
TODO: set DRAW_PROBABILITY in a more principled way. Beware twiddling it haphazardly, however, since it can mess with
existing ratings.
"""
DRAW_PROBABILITY = 0.1


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


class Agent:
    def __init__(self, cmd: str, rowid: Optional[int] = None, special: bool = False):
        """
        cmd looks like:

        <binary> --player "--type=MCTS-C --nnet-filename /media/dshin/c4f/models/gen-10.ptj"
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
        self.rating = trueskill.Rating()
        self.rowid = rowid
        self.special = special

    def get_cmd(self, binary_kwargs: Optional[dict] = None, player_kwargs: Optional[dict] = None):
        cmd = self.binary
        if binary_kwargs:
            for key, value in binary_kwargs.items():
                assert key.startswith('-'), key
                cmd += f' {key} {value}'

        player_str = self.player_str
        if player_kwargs:
            for key, value in player_kwargs.items():
                player_str = inject_arg(player_str, key, str(value))

        cmd += f' --player "{player_str}"'
        return cmd


class Arena:
    def __init__(self):
        self.c4_base_dir = os.path.join(Args.c4_base_dir_root, Args.tag)
        assert os.path.isdir(self.c4_base_dir), self.c4_base_dir
        self.arena_dir = os.path.join(self.c4_base_dir, 'arena')
        os.makedirs(self.arena_dir, exist_ok=True)

        self.all_agents = []
        self.special_agents = []
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
                special);
            """)
            c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS lookup ON agents (cmd);""")

            players_dir = os.path.join(self.c4_base_dir, 'players')
            for filename in natsorted(os.listdir(players_dir)):  # gen-13.txt
                if filename.startswith('.') or not filename.endswith('.txt'):
                    continue

                gen = int(filename.split('.')[0].split('-')[1])
                with open(os.path.join(players_dir, filename)) as f:
                    cmd = f.read().strip()

                cmd_tuples = [(cmd, False)]
                if gen == 0:
                    agent = Agent(cmd)
                    rand_cmd = f'{agent.binary} --player "--type=Random"'
                    perfect_cmd = f'{agent.binary} --player "--type=Perfect"'
                    cmd4 = agent.get_cmd(player_kwargs={'-i': 4})
                    cmd16 = agent.get_cmd(player_kwargs={'-i': 16})
                    cmd64 = agent.get_cmd(player_kwargs={'-i': 64})
                    aux_cmds = [rand_cmd, cmd4, cmd16, cmd64]
                    aux_cmd_tuples = [(cmd, True) for cmd in aux_cmds]
                    cmd_tuples = aux_cmd_tuples + [(perfect_cmd, False)] + cmd_tuples

                c.executemany('INSERT INTO agents VALUES (?, ?)', cmd_tuples)

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
            mu REAL, 
            sigma REAL);
        """)
        self.conn.commit()

    def init_agents(self):
        c = self.conn.cursor()
        res = c.execute('SELECT rowid, cmd, special FROM agents')
        for rowid, cmd, special in res.fetchall():
            agent = Agent(cmd, rowid, special)
            self.all_agents.append(agent)
            if special:
                self.special_agents.append(agent)
        timed_print(f'Loaded {len(self.all_agents)} agents (including {len(self.special_agents)} special agents)')

    def launch(self):
        trueskill.setup(tau=0, draw_probability=DRAW_PROBABILITY, backend='scipy')
        self.init_db()
        self.init_agents()

        self.play_special_round()
        for round in range(Args.num_rounds):
            timed_print(f'Round {round}')
            self.play_round()
            self.update_ratings()

    def play_special_round(self):
        for agent in self.special_agents:
            opponent = agent
            while opponent is agent:
                opponent = random.choice(self.all_agents)
            self.play_match(agent, opponent)

    def play_round(self):
        for agent in self.all_agents:
            opponent = agent
            while opponent is agent:
                opponent = random.choice(self.all_agents)
            self.play_match(agent, opponent)

    def play_match(self, agent1: Agent, agent2: Agent):
        binary_kwargs = {'-G': Args.n_games}

        player1_kwargs = {'--name': 'P1'}
        if not agent1.special:
            player1_kwargs['-i'] = Args.mcts_iters

        player2_kwargs = {'--name': 'P2'}
        if not agent2.special:
            player2_kwargs['-i'] = Args.mcts_iters

        if agent1.binary == agent2.binary:
            # might as well run both in the same process
            cmd = agent1.get_cmd(binary_kwargs=binary_kwargs, player_kwargs=player1_kwargs)
            player_str2 = inject_args(agent2.player_str, player2_kwargs)
            cmd += f' --player "{player_str2}"'
            print('cmd: ' + cmd)
        else:
            port = 12345
            binary_kwargs['--port'] = port
            cmd1 = agent1.get_cmd(binary_kwargs=binary_kwargs, player_kwargs=player1_kwargs)
            cmd2 = agent2.get_cmd(binary_kwargs={'--remote-port': port}, player_kwargs=player2_kwargs)
            print('cmd1: ' + cmd1)
            print('cmd2: ' + cmd2)

    def update_ratings(self):
        print('TODO: update ratings')


def main():
    load_args()
    arena = Arena()
    arena.launch()


if __name__ == '__main__':
    main()
