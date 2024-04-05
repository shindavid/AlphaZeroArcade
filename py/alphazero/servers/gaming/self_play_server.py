from alphazero.logic.custom_types import ClientRole
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.log_forwarder import LogForwarder
from alphazero.servers.gaming.session_data import SessionData
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util import subprocess_util

from dataclasses import dataclass, fields
import logging
import threading


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
    def __init__(self, params: SelfPlayServerParams, logging_params: LoggingParams):
        self._params = params
        self._session_data = SessionData(params)
        self._shutdown_manager = ShutdownManager()
        self._log_forwarder = LogForwarder(self._shutdown_manager, logging_params)
        self._running = False

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
        self._log_forwarder.set_socket(self._session_data.socket)
        self._shutdown_manager.register(lambda: self._session_data.socket.close())

    def _send_handshake(self):
        self._session_data.send_handshake(ClientRole.SELF_PLAY_SERVER)

    def _recv_handshake(self):
        self._session_data.recv_handshake(ClientRole.SELF_PLAY_SERVER, self._log_forwarder)

    def _recv_loop(self):
        try:
            self._send_ready()
            while True:
                msg = self._session_data.socket.recv_json()
                if self._handle_msg(msg):
                    break
        except SocketRecvException:
            logger.warn('Encountered SocketRecvException in recv_loop(). '
                        'Loop controller likely shut down.')
            self._shutdown_manager.request_shutdown(0)
        except SocketSendException:
            # Include exc_info in send-case because it's a bit more unexpected
            logger.warn('Encountered SocketSendException in recv_loop(). '
                        'Loop controller likely shut down.', exc_info=True)
            self._shutdown_manager.request_shutdown(0)
        except:
            logger.error(f'Unexpected error in recv_loop():', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-server received json message: {msg}')

        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self._handle_start_gen0(msg)
        elif msg_type == 'start':
            self._handle_start()
        elif msg_type == 'quit':
            self._quit()
            return True
        else:
            raise Exception(f'Unknown message type: {msg_type}')
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

    def _quit(self):
        logger.info(f'Received quit command')
        self._shutdown_manager.request_shutdown(0)

    def _start_gen0(self, msg: JsonDict):
        try:
            self._start_gen0_helper(msg)
        except:
            logger.error(f'Error in start_gen0:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _start_gen0_helper(self, msg):
        assert not self._running
        self._running = True

        max_rows = msg['max_rows']

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--max-rows', max_rows,
            '--no-model',

            # for gen-0, sample more positions and use fewer iters per game, so we finish faster
            '--num-full-iters', 100,
            '--full-pct', 1.0,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self._session_data.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self._params.loop_controller_host,
            '--loop-controller-port', self._params.loop_controller_port,
            '--client-role', ClientRole.SELF_PLAY_WORKER.value,
            '--do-not-report-metrics',
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running gen-0 self-play [{proc.pid}]: {self_play_cmd}')
        self._log_forwarder.forward_output('gen0-self-play-worker', proc)

        logger.info(f'Gen-0 self-play complete!')
        self._running = False

        data = {
            'type': 'gen0-complete',
        }
        self._session_data.socket.send_json(data)

    def _start(self):
        try:
            self._start_helper()
        except:
            logger.error(f'Error in start:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _start_helper(self):
        assert not self._running
        self._running = True

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--cuda-device', self._params.cuda_device,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self._session_data.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self._params.loop_controller_host,
            '--loop-controller-port', self._params.loop_controller_port,
            '--client-role', ClientRole.SELF_PLAY_WORKER.value,
            '--cuda-device', self._params.cuda_device,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')
        self._log_forwarder.forward_output('self-play-worker', proc)
        assert False, 'Should not get here'
