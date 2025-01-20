from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData
from util.logging_util import LoggingParams, get_logger
from util.py_util import register_signal_exception
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util.str_util import make_args_str
from util import subprocess_util

from dataclasses import dataclass, fields
import signal
import subprocess
import threading
from typing import Optional


logger = get_logger()


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
        self._session_data = SessionData(params, logging_params)
        self._shutdown_manager = ShutdownManager()
        self._running = False
        self._proc: Optional[subprocess.Popen] = None

        register_signal_exception(signal.SIGTERM)
        if params.ignore_sigint:
            signal.signal(signal.SIGINT, signal.SIG_IGN)

    def run(self):
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self._shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('Caught Ctrl-C')
        finally:
            self._shutdown_manager.shutdown()

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
        self._shutdown_manager.register(lambda: self._session_data.socket.close())

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
        elif msg_type == 'start':
            self._handle_start()
        elif msg_type == 'restart':
            self._handle_restart()
        elif msg_type == 'quit':
            self._quit()
            return True
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

    def _handle_start(self):
        thread = threading.Thread(target=self._start, daemon=True, name=f'start')
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
        assert not self._running
        self._running = True

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

        binary = self._build_params.get_binary_path(self._session_data.game)

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

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info('Running gen-0 self-play [%s]: %s', proc.pid, self_play_cmd)
        self._session_data.wait_for(proc)

        logger.info('Gen-0 self-play complete!')
        self._running = False

        data = {
            'type': 'gen0-complete',
        }
        self._session_data.socket.send_json(data)

        self._session_data.stop_log_sync(log_filename)

    def _start(self):
        try:
            self._start_helper()
        except:
            logger.error('Error in start:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _start_helper(self):
        assert not self._running
        self._running = True

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

        binary = self._build_params.get_binary_path(self._session_data.game)

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

        proc = subprocess_util.Popen(self_play_cmd)
        self._proc = proc
        logger.info('Running self-play [%s]: %s', proc.pid, self_play_cmd)
        self._session_data.wait_for(proc)

    def _restart_helper(self):
        proc = self._proc
        assert proc is not None

        logger.info('Restarting self-play process')
        logger.info('Killing [%s]...', proc.pid)
        self._session_data.disable_next_returncode_check()
        self._proc.kill()
        self._proc.wait(timeout=60)  # overly generous timeout, kill should be quick
        self._running = False

        self._start_helper()
