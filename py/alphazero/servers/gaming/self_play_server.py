from alphazero.logic.custom_types import ClientRole
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.game_server_base import GameServerBase
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict
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


class SelfPlayServer(GameServerBase):
    def __init__(self, params: SelfPlayServerParams, logging_params: LoggingParams):
        super().__init__(params, logging_params, ClientRole.SELF_PLAY_SERVER)
        self._running = False

    def run(self):
        try:
            threading.Thread(target=self._main_loop, name='main_loop', daemon=True).start()
            self.shutdown_manager.wait_for_shutdown_request()
        except KeyboardInterrupt:
            logger.info('Caught Ctrl-C')
        finally:
            self.shutdown_manager.shutdown()

    def _main_loop(self):
        try:
            self.init_socket()
            self.shutdown_manager.register(lambda: self.loop_controller_socket.close())
            self.send_handshake()
            self.recv_handshake()

            threading.Thread(target=self.recv_loop, daemon=True).start()
        except:
            logger.error('Unexpected error in main_loop():', exc_info=True)
            self.shutdown_manager.request_shutdown(1)

    def handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-server received json message: {msg}')

        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self._handle_start_gen0(msg)
        elif msg_type == 'start':
            self._handle_start()
        elif msg_type == 'quit':
            self.quit()
            return True
        else:
            raise Exception(f'Unknown message type: {msg_type}')
        return False

    def recv_loop_prelude(self):
        data = {
            'type': 'ready',
        }
        self.loop_controller_socket.send_json(data)

    def _handle_start_gen0(self, msg: JsonDict):
        thread = threading.Thread(target=self._start_gen0, args=(msg,),
                                  daemon=True, name=f'start-gen0')
        thread.start()

    def _start_gen0(self, msg: JsonDict):
        try:
            self._start_gen0_helper(msg)
        except:
            logger.error(f'Error in start_gen0:', exc_info=True)
            self.shutdown_manager.request_shutdown(1)

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
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientRole.SELF_PLAY_WORKER.value,
            '--do-not-report-metrics',
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running gen-0 self-play [{proc.pid}]: {self_play_cmd}')
        self.forward_output('gen0-self-play-worker', proc)

        logger.info(f'Gen-0 self-play complete!')
        self._running = False

        data = {
            'type': 'gen0-complete',
        }
        self.loop_controller_socket.send_json(data)

    def _handle_start(self):
        thread = threading.Thread(target=self._start, daemon=True, name=f'start')
        thread.start()

    def _start(self):
        try:
            self._start_helper()
        except:
            logger.error(f'Error in start:', exc_info=True)
            self.shutdown_manager.request_shutdown(1)

    def _start_helper(self):
        assert not self._running
        self._running = True

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--cuda-device', self.cuda_device,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientRole.SELF_PLAY_WORKER.value,
            '--cuda-device', self.cuda_device,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')
        self.forward_output('self-play-worker', proc)
        assert False, 'Should not get here'
