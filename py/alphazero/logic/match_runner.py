from alphazero.logic.agent_types import Agent, Match, MatchType
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util import subprocess_util
from util.str_util import make_args_str

from dataclasses import dataclass
from enum import Enum

import logging
from typing import Dict, Optional



logger = logging.getLogger(__name__)



class MatchRunner:
    @staticmethod
    def run_match_helper(match: Match, game: str, args: Optional[Dict]=None) -> WinLossDrawCounts:
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

        args1 = dict(args)
        args2 = dict(args)

        port = DEFAULT_REMOTE_PLAY_PORT

        cmd1 = [
            agent1.binary,
            '--port', str(port),
            '--player', f'"{ps1}"',
        ]
        cmd1.append(make_args_str(args1))
        cmd1 = ' '.join(map(str, cmd1))

        cmd2 = [
            agent2.binary,
            '--remote-port', str(port),
            '--player', f'"{ps2}"',
        ]
        cmd2.append(make_args_str(args2))
        cmd2 = ' '.join(map(str, cmd2))

        logger.debug('Running match between:\n%s\n%s', cmd1, cmd2)

        proc1 = subprocess_util.Popen(cmd1)
        proc2 = subprocess_util.Popen(cmd2)

        expected_rc = None
        print_fn = logger.error
        stdout = subprocess_util.wait_for(proc1, expected_return_code=expected_rc, print_fn=print_fn)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info(f'Match Result:\n{match.agent1} vs {match.agent2}: {record.get(0)}')
        return record.get(0)
