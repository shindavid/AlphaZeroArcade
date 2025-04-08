from alphazero.logic.agent_types import MCTSAgent
from alphazero.logic.build_params import BuildParams
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.base_server import BaseServer, ServerConstants
from util.logging_util import LoggingParams
from util.socket_util import JsonDict
from util import subprocess_util
from util.str_util import make_args_str

from dataclasses import dataclass, fields
import logging
import os
from typing import Optional, Dict


logger = logging.getLogger(__name__)


@dataclass
class EvalServerParams(BaseParams):
    rating_tag: str = ''

    @staticmethod
    def create(args) -> 'EvalServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(EvalServerParams)}
        return EvalServerParams(**kwargs)

    @staticmethod
    def add_args(parser, omit_base=False):
        defaults = EvalServerParams()

        group = parser.add_argument_group(f'EvalServer options')
        if not omit_base:
            BaseParams.add_base_args(group)

        group.add_argument('-r', '--rating-tag', default=defaults.rating_tag,
                           help='evaluation tag. Loop controller collates ratings by this str. It is '
                           'the responsibility of the user to make sure that the same '
                           'binary/params are used across different EvalServer processes '
                           'sharing the same rating-tag. (default: "%(default)s")')


class EvalServer(BaseServer):
    SERVER_CONSTANTS = ServerConstants(
        server_name='eval-server',
        worker_name='eval-worker',
        server_role=ClientRole.EVAL_SERVER,
        worker_role=ClientRole.EVAL_WORKER)

    def __init__(self, params: EvalServerParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        super().__init__(params, logging_params, build_params)

    def _send_handshake(self):
        additional_data = {'rating_tag': self._params.rating_tag}
        self._session_data.send_handshake(ClientRole.EVAL_SERVER,
                                          addtional_data=additional_data)

    def _run_match_helper(self, msg: JsonDict):
        assert not self._running
        self._running = True

        files_required = [FileToTransfer(**f) for f in msg['files_required']]
        self._session_data.request_files(files_required)

        mcts_agent1 = MCTSAgent(**msg['agent1'])
        mcts_agent2 = MCTSAgent(**msg['agent2'])
        match = Match(mcts_agent1, mcts_agent2, msg['n_games'], MatchType.EVALUATE)

        log_filename = self._session_data.get_log_filename('eval-worker')
        args = {
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': ClientRole.EVAL_WORKER.value,
            '--manager-id': self._session_data.client_id,
            '--ratings-tag': f'"{self._params.rating_tag}"',
            '--cuda-device': self._params.cuda_device,
            '--do-not-report-metrics': None,
            '--log-filename': log_filename,
        }
        result = self._eval_match(match, args)

        logger.info('Match result between:\n%s\n%s\nresult: %s', msg['agent1'], msg['agent2'], result)

        self._running = False

        data = {
            'type': 'match-result',
            'record': result.to_json(),
            'ix1': msg['ix1'],
            'ix2': msg['ix2']
        }

        self._session_data.socket.send_json(data)
        self._send_ready()

    def _eval_match(self, match: Match, args: Optional[Dict]=None) -> WinLossDrawCounts:
        """
        Run a match between two agents and return the results by running two subprocesses
        of C++ binaries.
        """
        agent1 = match.agent1
        agent2 = match.agent2
        n_games = match.n_games
        if n_games < 1:
            return WinLossDrawCounts()

        ps1 = agent1.make_player_str(self._session_data.run_dir)
        ps2 = agent2.make_player_str(self._session_data.run_dir)

        if args is None:
            args = {}
        args['-G'] = n_games

        args1 = dict(args)
        args2 = dict(args)

        port = DEFAULT_REMOTE_PLAY_PORT

        cmd1 = [
            os.path.join(self._session_data.run_dir, agent1.binary),
            '--port', str(port),
            '--player', f'"{ps1}"',
        ]
        cmd1.append(make_args_str(args1))
        cmd1 = ' '.join(map(str, cmd1))

        cmd2 = [
            os.path.join(self._session_data.run_dir, agent2.binary),
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
        return record.get(0)
