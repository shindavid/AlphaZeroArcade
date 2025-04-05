from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData
from util.logging_util import LoggingParams
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util.str_util import make_args_str
from util import subprocess_util

from dataclasses import dataclass, fields
import logging
import os
import subprocess
import threading
from typing import List, Optional


logger = logging.getLogger(__name__)


@dataclass
class SelfPlayServerParams(BaseParams):
    @staticmethod
    def create(args) -> 'SelfPlayServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(SelfPlayServerParams)}
        return SelfPlayServerParams(**kwargs)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group(f'SelfPlayServer options')
        BaseParams.add_base_args(group)


class SelfPlayServer:
    def __init__(self, params: SelfPlayServerParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        self._params = params
        self._build_params = build_params
        self._session_data = SessionData(params, logging_params, build_params)
        self._shutdown_manager = ShutdownManager()

        self._proc: Optional[subprocess.Popen] = None
        self._proc_cond = threading.Condition()

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
        logger.info('Shutting down self-play server...')
        try:
            self._session_data.socket.close()
        except:
            pass

        if self._proc is not None:
            try:
                self._proc.terminate()
                subprocess_util.wait_for(self._proc, expected_return_code=None)
                logger.info('Terminated ratings-worker process %s', self._proc.pid)
            except:
                pass
        logger.info('Self-play server shutdown complete!')

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
        self._session_data.send_handshake(ClientRole.SELF_PLAY_SERVER)

    def _recv_handshake(self):
        self._session_data.recv_handshake(ClientRole.SELF_PLAY_SERVER)

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

    def _handle_msg(self, msg: JsonDict) -> bool:
        logger.debug('self-play-server received json message: %s', msg)

        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self._handle_start_gen0(msg)
        elif msg_type == 'stop-gen0':
            self._handle_stop_gen0()
        elif msg_type == 'start':
            self._handle_start(msg)
        elif msg_type == 'restart':
            self._handle_restart()
        elif msg_type == 'quit':
            self._quit()
            return True
        elif msg_type == 'binary-file':
            self._session_data.receive_binary_file(msg['binary'])
        else:
            raise Exception('Unknown message type: %s', msg_type)
        return False

    def _send_ready(self):
        data = {'type': 'ready', }
        self._session_data.socket.send_json(data)

    def _handle_start_gen0(self, msg: JsonDict):
        thread = threading.Thread(target=self._start_gen0, args=(msg,),
                                  daemon=True, name=f'start-gen0')
        thread.start()

    def _handle_stop_gen0(self):
        # NOTE: although my intention is for this to be the point at which the process is killed,
        # the gen-0 c++ process seems to exit on its own. I'm not exactly sure why, but it doesn't
        # really matter, so I'm not going to investigate further.
        proc = self._proc
        assert proc is not None

        logger.info('Stopping gen-0 self-play process [%s]', proc.pid)
        self._session_data.disable_next_returncode_check()
        self._proc.kill()
        self._proc.wait(timeout=60)  # overly generous timeout, kill should be quick
        self._clear_proc()

    def _handle_start(self, msg: JsonDict):
        thread = threading.Thread(target=self._start, args=(msg,), daemon=True, name=f'start')
        thread.start()

    def _handle_restart(self):
        try:
            thread = threading.Thread(target=self._restart_helper, daemon=True, name=f'restart')
            thread.start()
        except:
            logger.error('Error in restart:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _quit(self):
        logger.info('Received quit command')
        self._shutdown_manager.request_shutdown(0)

    def _start_gen0(self, msg: JsonDict):
        try:
            self._start_gen0_helper(msg)
        except:
            logger.error('Error in start_gen0:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _start_gen0_helper(self, msg):
        required_binary = msg['binary']
        missing_binaries: List[FileToTransfer] = self._session_data.get_missing_binaries([required_binary])
        if missing_binaries:
            logger.warning('Missing required binaries: %s', missing_binaries)
            self._session_data.send_binary_request(missing_binaries)
            self._session_data.wait_for_binaries([required_binary])

        max_rows = msg['max_rows']

        player_args = {
            '--type': 'MCTS-T',
            '--name': 'MCTS',
            '--no-model': None,
        }
        player_args.update(self._session_data.game_spec.training_player_options)

        # for gen-0, sample more positions and use fewer iters per game, so we finish faster
        player_args.update({
            '-i': 100,
        })
        player_args_str = make_args_str(player_args)

        log_filename = self._session_data.get_log_filename('gen0-self-play-worker')
        self._session_data.start_log_sync(log_filename)

        args = {
            '-G': 0,
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': ClientRole.SELF_PLAY_WORKER.value,
            '--do-not-report-metrics': None,
            '--max-rows': max_rows,
            '--enable-training': None,
            '--log-filename': log_filename,
        }
        args.update(self._session_data.game_spec.training_options)

        if self._session_data.directory_organizer is not None:
            # Needed for direct-game-log-write optimization
            args['--output-base-dir'] = self._session_data.directory_organizer.base_dir

        binary_path = FileToTransfer(**required_binary).scratch_path
        binary = os.path.join(self._session_data.run_dir, binary_path)
        self_play_cmd = [
            binary,
            '--player', '"%s"' % player_args_str,
        ]
        for p in range(self._session_data.game_spec.num_players - 1):
            opp_args = {
                '--name': 'MCTS%d' % (p + 2),
                '--copy-from': 'MCTS',
            }
            opp_args_str = make_args_str(opp_args)
            self_play_cmd.append('--player')
            self_play_cmd.append(f'"{opp_args_str}"')

        self_play_cmd.append(make_args_str(args))
        self_play_cmd = ' '.join(map(str, self_play_cmd))

        cwd = self._session_data.run_dir
        proc = self._set_proc(self_play_cmd, cwd)
        logger.info('Running gen-0 self-play [%s] from %s: %s', proc.pid, cwd, self_play_cmd)
        self._session_data.wait_for(proc)
        self._clear_proc()

        logger.info('Gen-0 self-play complete!')
        self._session_data.stop_log_sync(log_filename)

    def _start(self, msg: JsonDict):
        try:
            self._start_helper(msg)
        except:
            logger.error('Error in start:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _start_helper(self, msg: JsonDict):
        required_binary = msg['binary']
        missing_binaries: List[FileToTransfer] = self._session_data.get_missing_binaries([required_binary])
        if missing_binaries:
            logger.warning('Missing required binaries: %s', missing_binaries)
            self._session_data.send_binary_request(missing_binaries)
            self._session_data.wait_for_binaries([required_binary])

        player_args = {
            '--type': 'MCTS-T',
            '--name': 'MCTS',
            '--cuda-device': self._params.cuda_device,
        }
        player_args.update(self._session_data.game_spec.training_player_options)
        player_args_str = make_args_str(player_args)

        log_filename = self._session_data.get_log_filename('self-play-worker')
        self._session_data.start_log_sync(log_filename)

        args = {
            '-G': 0,
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': ClientRole.SELF_PLAY_WORKER.value,
            '--cuda-device': self._params.cuda_device,
            '--enable-training': None,
            '--log-filename': log_filename,
        }
        args.update(self._session_data.game_spec.training_options)

        if self._session_data.directory_organizer is not None:
            # Needed for direct-game-log-write optimization
            args['--output-base-dir'] = self._session_data.directory_organizer.base_dir

        binary_path = FileToTransfer(**required_binary).scratch_path
        binary = os.path.join(self._session_data.run_dir, binary_path)
        self_play_cmd = [
            binary,
            '--player', '"%s"' % player_args_str,
        ]
        for p in range(self._session_data.game_spec.num_players - 1):
            opp_args = {
                '--name': 'MCTS%d' % (p + 2),
                '--copy-from': 'MCTS',
            }
            opp_args_str = make_args_str(opp_args)
            self_play_cmd.append('--player')
            self_play_cmd.append(f'"{opp_args_str}"')

        self_play_cmd.append(make_args_str(args))
        self_play_cmd = ' '.join(map(str, self_play_cmd))

        cwd = self._session_data.run_dir
        proc = self._set_proc(self_play_cmd, cwd)
        logger.info('Running self-play [%s] from %s: %s', proc.pid, cwd, self_play_cmd)
        self._session_data.wait_for(proc)
        self._clear_proc()

    def _restart_helper(self):
        proc = self._proc
        assert proc is not None

        logger.info('Restarting self-play process')
        logger.info('Killing [%s]...', proc.pid)
        self._session_data.disable_next_returncode_check()
        self._proc.kill()
        self._proc.wait(timeout=60)  # overly generous timeout, kill should be quick
        self._clear_proc()

        self._start_helper()

    def _set_proc(self, cmd: List[str], cwd: str) -> subprocess.Popen:
        with self._proc_cond:
            self._proc_cond.wait_for(lambda: self._proc is None)
            proc = subprocess_util.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL)
            self._proc = proc
            return proc

    def _clear_proc(self):
        with self._proc_cond:
            self._proc = None
            self._proc_cond.notify_all()
