from alphazero.logic.agent_types import Agent, MCTSAgent, PerfectAgent, UniformAgent
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import subprocess_util
from util.logging_util import get_logger
from util.str_util import make_args_str

from dataclasses import dataclass
from itertools import combinations, product

logger = get_logger()

@dataclass
class Match:
    agent1: Agent
    agent2: Agent
    n_games: int

class MatchRunner:
    @staticmethod
    def linspace_matches(gen_start: int, gen_end: int, n_iters: int=100, freq:int =1,\
      n_games: int =100, organizer: DirectoryOrganizer=None):
        gens = range(gen_start, gen_end + 1, freq)
        matches = []
        for gen1, gen2 in combinations(gens, 2):
            if gen1 == 0:
                agent1 = UniformAgent(n_iters=n_iters)
            else:
                agent1 = MCTSAgent(gen=gen1, n_iters=n_iters, organizer=organizer)

            if gen2 == 0:
                agent2 = UniformAgent(n_iters=n_iters)
            else:
                agent2 = MCTSAgent(gen=gen2, n_iters=n_iters, organizer=organizer)
            match = Match(agent1=agent1, agent2=agent2, n_games=n_games)
            matches.append(match)
        return matches

    @staticmethod
    def run_match_helper(match: Match, binary):
        agent1 = match.agent1
        agent2 = match.agent2
        n_games = match.n_games
        if n_games < 1:
            return WinLossDrawCounts()

        ps1 = agent1.make_player_str(set_temp_zero=True)
        ps2 = agent2.make_player_str(set_temp_zero=True)

        base_args = {
            '-G': n_games,
            '--do-not-report-metrics': None,
        }

        args1 = dict(base_args)
        args2 = dict(base_args)

        port = 1234  # TODO: move this to constants.py or somewhere

        cmd1 = [
            binary,
            '--port', str(port),
            '--player', f'"{ps1}"',
        ]
        cmd1.append(make_args_str(args1))
        cmd1 = ' '.join(map(str, cmd1))

        cmd2 = [
            binary,
            '--remote-port', str(port),
            '--player', f'"{ps2}"',
        ]
        cmd2.append(make_args_str(args2))
        cmd2 = ' '.join(map(str, cmd2))

        proc1 = subprocess_util.Popen(cmd1)
        proc2 = subprocess_util.Popen(cmd2)

        expected_rc = None
        print_fn = logger.error
        stdout = subprocess_util.wait_for(proc1, expected_return_code=expected_rc, print_fn=print_fn)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info('Match result: %s', record.get(0))
        return record.get(0)
