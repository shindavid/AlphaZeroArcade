from alphazero.logic.agent_types import Agent, MCTSAgent, ReferenceAgent
from alphazero.logic.build_params import BuildParams
from alphazero.logic.constants import DEFAULT_REMOTE_PLAY_PORT
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.match_runner import Match, MatchType
from alphazero.logic.ratings import WinLossDrawCounts, extract_match_record
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from alphazero.servers.gaming import platform_overrides
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData
from shared.rating_params import RatingParams
from util import subprocess_util
from util.logging_util import LoggingParams
from util.py_util import sha256sum
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util.str_util import make_args_str

from dataclasses import dataclass, fields
import os
import logging
import threading
from typing import Dict, Optional, Set
import subprocess


logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    server_name: str
    worker_name: str
    server_role: ClientRole
    worker_role: ClientRole


@dataclass
class ServerParams(BaseParams):
    @classmethod
    def create(cls, args) -> 'ServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def add_args(cls, parser, omit_base=False, server_name: Optional[str]=None):
        group_title = f'{cls.__name__} options'
        group = parser.add_argument_group(group_title)

        if not omit_base:
            BaseParams.add_base_args(group)

        cls.add_additional_args(group)

    @staticmethod
    def add_additional_args(group):
        pass


class ServerBase:
    def __init__(self, params: ServerParams, logging_params: LoggingParams,
                 build_params: BuildParams, rating_params: RatingParams, server_config: ServerConfig):
        self._params = params
        self._build_params = build_params
        self._rating_params = rating_params
        self._config = server_config
        self._session_data = SessionData(params, logging_params, build_params)
        self._shutdown_manager = ShutdownManager()
        self._running = False
        self._shutdown_manager.register(self._shutdown, 'server-base-shutdown')
        self._procs: Set[subprocess.Popen] = set()
        register_standard_server_signals(ignore_sigint=params.ignore_sigint)

    def run(self):
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('server_base Caught Ctrl-C')
        except SystemExit:
            logger.info('server_base caught SystemExit')
        finally:
            self._shutdown_manager.shutdown()

    def _shutdown(self):
        logger.info('Shutting down %s...', self._config.server_name)
        try:
            self._session_data.socket.close()
        except:
            pass

        subprocess_util.terminate_processes(self._procs)
        self._procs.clear()

        logger.info('%s server shutdown complete!', self._config.server_name)

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
        self._session_data.send_handshake(self._config.server_role)

    def _recv_handshake(self):
        self._session_data.recv_handshake(self._config.server_role)

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
        logger.debug('%s received json message: %s', self._config.server_name, msg)

        msg_type = msg['type']
        if msg_type == 'match-request':
            self._handle_match_request(msg)
        elif msg_type == 'file-transfer':
            self._session_data.receive_files(msg['files'])
        elif msg_type == 'quit':
            self._quit()
            return True
        else:
            raise Exception('Unknown message type: %s', msg_type)
        return False

    def _handle_match_request(self, msg: JsonDict):
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

        files_required = [FileToTransfer(**f) for f in msg['files_required']]
        self._session_data.request_files(files_required)

        agent1 = self._build_agent(msg['agent1'])
        agent2 = self._build_agent(msg['agent2'])
        match = Match(agent1, agent2, msg['n_games'], MatchType.EVALUATE)

        args = {
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': self._config.worker_role.value,
            '--manager-id': self._session_data.client_id,
            '--cuda-device': self._params.cuda_device,
            '--do-not-report-metrics': None,
        }

        platform_overrides.update_cpp_bin_args(args)
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

        num_thread_option = {'-n': self.num_threads}
        run_dir = self._session_data.run_dir

        ps1 = agent1.make_player_str(run_dir, args=num_thread_option, suffix='-A')
        ps2 = agent2.make_player_str(run_dir, args=num_thread_option, suffix='-B')
        binary1 = os.path.join(run_dir, agent1.binary)
        binary2 = os.path.join(run_dir, agent2.binary)

        if args is None:
            args = {}

        if self._rating_params.use_remote_play:
            port = DEFAULT_REMOTE_PLAY_PORT
            log_filename1 = self._session_data.get_log_filename(self._config.worker_name + '-A')
            log_filename2 = self._session_data.get_log_filename(self._config.worker_name + '-B')

            cmd1 = [
                binary1,
                '--port', str(port),
                '--player', f'"{ps1}"',
                '--log-filename', log_filename1,
                '-G', str(n_games),
                ]

            if not self._session_data.start_log_sync(log_filename1):
                cmd1.append('--log-append-mode')

            # If two or more MCTS agents are involved, enable alternating mode
            mcts_count = 0
            for agent in (agent1, agent2):
                if isinstance(agent, MCTSAgent):
                    mcts_count += 1

            if mcts_count > 1:
                cmd1.append('--alternating-mode=2')

            cmd1.append(make_args_str(args))
            cmd1 = ' '.join(map(str, cmd1))

            cmd2 = [
                binary2,
                '--remote-port', str(port),
                '--player', f'"{ps2}"',
                '--log-filename', log_filename2,
            ]
            if not self._session_data.start_log_sync(log_filename2):
                cmd2.append('--log-append-mode')
            cmd2.append(make_args_str(args))
            cmd2 = ' '.join(map(str, cmd2))

            logger.info('Running match between:%s vs %s', agent1, agent2)
            logger.info('cmd1: %s', cmd1)
            logger.info('cmd2: %s', cmd2)

            proc1 = subprocess_util.Popen(cmd1, cwd=run_dir)
            proc2 = subprocess_util.Popen(cmd2, cwd=run_dir)

            procs_list = [proc1, proc2]
        else:
            if sha256sum(binary1) != sha256sum(binary2):
                raise Exception(
                    f'Binary mismatch: {binary1} and {binary2} are not the same when use-remote-play is False.'
                    'Please use the same binary for both players or set use-remote-play to True.'                )

            log_filename = self._session_data.get_log_filename(self._config.worker_name)
            cmd = [binary1,
                   '--player', f'"{ps1}"',
                   '--player', f'"{ps2}"',
                   '--log-filename', log_filename,
                   '-G', str(n_games),
                   ]
            if not self._session_data.start_log_sync(log_filename):
                cmd.append('--log-append-mode')
            cmd.append(make_args_str(args))
            cmd = ' '.join(map(str, cmd))
            logger.info('cmd:\n %s', cmd)
            proc1 = subprocess_util.Popen(cmd, cwd=run_dir)
            procs_list = [proc1]

        procs_set = set(procs_list)
        self._procs.update(procs_set)
        print_fn = logger.error
        stdout = subprocess_util.wait_for(procs_list, print_fn=print_fn)[0]

        self._procs.difference_update(procs_set)
        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        return record.get(0)

    @property
    def num_threads(self) -> int:
        return self._rating_params.rating_player_options.num_search_threads

    def _build_agent(self, agent_msg: JsonDict) -> Agent:
        if agent_msg['type'] == 'MCTS':
            return MCTSAgent(**agent_msg['data'])
        elif agent_msg['type'] == 'Reference':
            return ReferenceAgent(**agent_msg['data'])
        else:
            raise ValueError(f'Unknown agent type: {agent_msg["type"]}')
