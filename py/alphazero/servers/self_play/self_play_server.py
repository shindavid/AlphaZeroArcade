from alphazero.logic.custom_types import ClientRole
from alphazero.servers.game_server_base import GameServerBase, GameServerBaseParams
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict
from util import subprocess_util

from dataclasses import dataclass
import logging


logger = get_logger()


@dataclass
class SelfPlayServerParams(GameServerBaseParams):
    @staticmethod
    def add_args(parser):
        GameServerBaseParams.add_args_helper(parser, 'SelfPlayServer')


class SelfPlayServer(GameServerBase):
    def __init__(self, params: SelfPlayServerParams, logging_params: LoggingParams):
        super().__init__(params, logging_params, ClientRole.SELF_PLAY_SERVER)
        self._running = False

    def handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-server received json message: {msg}')

        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self.run_func_in_new_thread(self.start_gen0, args=(msg,))
        elif msg_type == 'start':
            self.run_func_in_new_thread(self.start, args=(msg,))
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

    def start_gen0(self, msg):
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

    def start(self, msg):
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
