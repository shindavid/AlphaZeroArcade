from alphazero.logic.agent_types import MCTSAgent
from alphazero.logic.build_params import BuildParams
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.logic.run_params import RunParams
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData, ASSETS_DIR
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util import subprocess_util
from util.str_util import make_args_str

from dataclasses import dataclass, fields
import subprocess
import threading
import os
from typing import Optional, List, Dict


logger = get_logger()


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


class EvalServer:
    def __init__(self, params: EvalServerParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        self._params = params
        self._build_params = build_params
        self._session_data = SessionData(params, logging_params, build_params)
        self._shutdown_manager = ShutdownManager()
        self._running = False
        self._proc: Optional[subprocess.Popen] = None

        self._shutdown_manager.register(lambda: self._shutdown())
        register_standard_server_signals(ignore_sigint=params.ignore_sigint)

    def run(self):
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('Caught Ctrl-C')
        finally:
            self._shutdown_manager.shutdown()

    def _shutdown(self):
        logger.info('Shutting down eval-server...')
        try:
            self._session_data.socket.close()
        except:
            pass

        if self._proc is not None:
            try:
                self._proc.terminate()
                subprocess_util.wait_for(self._proc, expected_return_code=None)
                logger.info('Terminated eval-worker process %s', self._proc.pid)
            except:
                pass
        logger.info('Eval server shutdown complete!')

    def _main_loop(self):
        try:
            self._init_socket()
            self._send_handshake()
            self._recv_handshake()

            threading.Thread(target=self._recv_loop, daemon=True).start()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _init_socket(self):
        self._session_data.init_socket()

    def _send_handshake(self):
        self._session_data.send_handshake(ClientRole.EVAL_SERVER,
                                    rating_tag=self._params.rating_tag)

    def _recv_handshake(self):
        self._session_data.recv_handshake(ClientRole.EVAL_SERVER)

    def _recv_loop(self):
        try:
            self._send_ready()
            while True:
                msg = self._session_data.socket.recv_json()
                if self._handle_msg(msg):
                    break
        except SocketRecvException:
            logger.warning('Encountered SocketRecvException in recv_loop(). '
                        'Loop controller likely shut down.')
            self._shutdown_manager.request_shutdown(0)
        except SocketSendException:
            # Include exc_info in send-case because it's a bit more unexpected
            logger.warning('Encountered SocketSendException in recv_loop(). '
                        'Loop controller likely shut down.', exc_info=True)
            self._shutdown_manager.request_shutdown(0)
        except:
            logger.error('Unexpected error in recv_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _send_ready(self):
        data = { 'type': 'ready', }
        self._session_data.socket.send_json(data)

    def _handle_msg(self, msg: JsonDict) -> bool:
        logger.debug('eval-server received json message: %s', msg)

        msg_type = msg['type']
        if msg_type == 'match-request':
            self._handle_match_request(msg)
        elif msg_type == 'binary-file':
            self._session_data.receive_binary_file(msg['binary'])
        elif msg_type == 'model-file':
            self._session_data.receive_model_file(msg['model'])
        elif msg_type == 'quit':
            self._quit()
            return True
        else:
            raise Exception('Unknown message type: %s', msg_type)
        return False

    def _handle_match_request(self, msg: JsonDict):
        logger.info('Received match request between gen %s and gen %s', msg['agent1']['gen'], msg['agent2']['gen'])
        thread = threading.Thread(target=self._run_match, args=(msg,), daemon=True,
                                  name='run-match')
        thread.start()

    def _quit(self):
        logger.info('Received quit command')
        self._shutdown_manager.request_shutdown(0)

    def _run_match(self, msg: JsonDict):
        try:
            self._run_match_helper(msg)
        except:
            logger.error('Unexpected error in run-match:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _run_match_helper(self, msg: JsonDict):
        assert not self._running
        self._running = True

        required_binaries = msg['binaries']

        missing_binaries: List[FileToTransfer] = self._session_data.get_missing_binaries(required_binaries)
        if missing_binaries:
            logger.warning('Missing required binaries: %s', missing_binaries)
            self._session_data.send_binary_request(missing_binaries)
            self._session_data.wait_for_binaries(required_binaries)

        required_model = msg['model_file']
        missing_model: bool = self._session_data.is_model_missing(required_model)
        if missing_model:
            logger.warning('Missing required model file: %s', required_model)
            self._session_data.send_model_request(required_model)
            self._session_data.wait_for_model(required_model)

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
        result = self._eval_match(match, self._session_data.game, args)

        logger.info('Played match between:\n%s\n%s\nresult: %s', msg['agent1'], msg['agent2'], result)

        self._running = False

        data = {
            'type': 'match-result',
            'record': result.to_json(),
            'ix1': msg['ix1'],
            'ix2': msg['ix2']
        }

        self._session_data.socket.send_json(data)
        self._send_ready()

    def _eval_match(self, match: Match, game: str, args: Optional[Dict]=None) -> WinLossDrawCounts:
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
            os.path.join(self._session_data.run_dir, agent1.binary),
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
        logger.info(f'{match.agent1} vs {match.agent2}: {record.get(0)}')
        return record.get(0)
