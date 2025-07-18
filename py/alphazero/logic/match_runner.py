from alphazero.logic.agent_types import Match
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.servers.gaming import platform_overrides
from util import subprocess_util
from util.str_util import make_args_str

import logging
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class MatchRunner:
    @staticmethod
    def run_match_helper(match: Match, binary: str, args: Optional[Dict] = None)\
            -> WinLossDrawCounts:
        """
        Run a match between two agents and return the results by running two subprocesses
        of C++ binaries.
        """
        logger.debug('Running match: %s vs %s', match.agent1, match.agent2)
        agent1 = match.agent1
        agent2 = match.agent2
        n_games = match.n_games
        if n_games < 1:
            return WinLossDrawCounts()

        ps1 = agent1.make_player_str('')
        ps2 = agent2.make_player_str('')

        if args is None:
            args = {}
        args['-G'] = n_games
        platform_overrides.update_cpp_bin_args(args)

        cmd = [
            binary,
            '--player', f'"{ps1}"',
            '--player', f'"{ps2}"',
        ]
        cmd.append(make_args_str(args))
        cmd = ' '.join(map(str, cmd))

        logger.info('Running match:\n%s', cmd)

        proc = subprocess_util.Popen(cmd)

        expected_rc = None
        print_fn = logger.error
        stdout = subprocess_util.wait_for(proc, expected_return_code=expected_rc, print_fn=print_fn)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info(f'Match Result:\n{match.agent1} vs {match.agent2}: {record.get(0)}')
        return record.get(0)
