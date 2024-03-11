from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ClientType
from alphazero.logic.game_server_base import GameServerBase, GameServerBaseParams
from util.logging_util import get_logger
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
    def __init__(self, params: SelfPlayServerParams, common_params: CommonParams):
        super().__init__(params, common_params, ClientType.SELF_PLAY_MANAGER)

    def handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'self-play-manager received json message: {msg}')

        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self.run_func_in_new_thread(self.start_gen0, args=(msg,))
        elif msg_type == 'start':
            self.start(msg)
        elif msg_type == 'quit':
            self.quit()
            return True
        return False

    def start_gen0(self, msg):
        assert self.child_process is None

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

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientType.SELF_PLAY_WORKER.value,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play.log')
        with open(log_filename, 'a') as log_file:
            logger.info(f'Running gen-0 self-play: {self_play_cmd}')
            proc = subprocess_util.Popen(self_play_cmd, stdout=log_file, stderr=subprocess.STDOUT)
            self.child_process = proc
            proc.wait()
        if proc.returncode:
            raise Exception(f'Gen-0 self-play failed with return code {proc.returncode}')
        self.child_process = None
        logger.info(f'Gen-0 self-play complete!')

        data = {
            'type': 'gen0-complete',
        }
        self.loop_controller_socket.send_json(data)

    def start(self, msg):
        assert self.child_process is None

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

        self_play_cmd = [
            self.binary_path,
            '-G', 0,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientType.SELF_PLAY_WORKER.value,
            '--starting-generation', gen,
            '--cuda-device', self.cuda_device,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play.log')
        log_file = open(log_filename, 'a')
        proc = subprocess_util.Popen(self_play_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        self.child_process = proc
        logger.info(f'Running self-play [{proc.pid}]: {self_play_cmd}')
