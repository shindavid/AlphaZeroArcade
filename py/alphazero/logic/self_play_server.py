from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ClientType
from alphazero.logic.game_server_base import GameServerBase, GameServerBaseParams
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict
from util import subprocess_util

from dataclasses import dataclass
import logging
import os
import subprocess


logger = get_logger()


@dataclass
class SelfPlayServerParams(GameServerBaseParams):
    @staticmethod
    def add_args(parser):
        GameServerBaseParams.add_args_helper(parser, 'SelfPlayServer')


class SelfPlayServer(GameServerBase):
    def __init__(self, params: SelfPlayServerParams, common_params: CommonParams,
                 logging_params: LoggingParams):
        super().__init__(params, common_params, logging_params, ClientType.SELF_PLAY_MANAGER)
        self._running = False

    def handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-manager received json message: {msg}')

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

    def start_gen0(self, msg):
        assert not self._running
        self._running = True

        # TODO: once we change c++ to directly communicate game data to the loop-controller via TCP,
        # we will no longer need games_base_dir here
        games_base_dir = msg['games_base_dir']
        max_rows = msg['max_rows']
        gen = 0

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--games-base-dir', games_base_dir,
            '--do-not-report-metrics',
            '--max-rows', max_rows,

            # for gen-0, sample more positions and use fewer iters per game, so we finish faster
            '--num-full-iters', 100,
            '--full-pct', 1.0,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        log_filename = self.make_log_filename('self-play-worker-gen0')
        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientType.SELF_PLAY_WORKER.value,
            '--starting-generation', gen,
            '--log-filename', log_filename,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running gen-0 self-play [{proc.pid}]: {self_play_cmd}')
        logger.info(f'Log: {log_filename}')
        _, stderr = proc.communicate()

        if proc.returncode:
            logger.error(f'Gen-0 self-play failed with return code {proc.returncode}')
            logger.error(f'stderr:\n{stderr}')
            raise Exception()

        logger.info(f'Gen-0 self-play complete!')
        self._running = False

        data = {
            'type': 'gen0-complete',
        }
        self.loop_controller_socket.send_json(data)

    def start(self, msg):
        assert not self._running
        self._running = True

        # TODO: once we change c++ to directly communicate game data to the loop-controller via TCP,
        # we will no longer need games_base_dir or model here
        games_base_dir = msg['games_base_dir']
        gen = msg['gen']
        model = msg['model']

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '--games-base-dir', games_base_dir,
            '-m', model,
            '--cuda-device', self.cuda_device,
        ]

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        log_filename = self.make_log_filename('self-play-worker')
        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientType.SELF_PLAY_WORKER.value,
            '--starting-generation', gen,
            '--cuda-device', self.cuda_device,
            '--log-filename', log_filename,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        proc = subprocess_util.Popen(self_play_cmd)
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')
        logger.info(f'Log: {log_filename}')
        _, stderr = proc.communicate()

        if proc.returncode:
            logger.error(f'Self-play failed with return code {proc.returncode}')
            logger.error(f'stderr:\n{stderr}')
            raise Exception()

        assert False, 'Should not get here'
