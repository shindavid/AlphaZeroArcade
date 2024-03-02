from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ClientType
from alphazero.logic.game_server_base import GameServerBase, GameServerBaseParams
from util.logging_util import get_logger
from util.socket_util import JsonDict
from util import subprocess_util

from dataclasses import dataclass
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
        super().__init__(params, common_params, ClientType.SELF_PLAY_SERVER)

    def handle_msg(self, msg: JsonDict) -> bool:
        msg_type = msg['type']
        if msg_type == 'start-gen0':
            self.start_gen0(msg)
        elif msg_type == 'start':
            self.start(msg)
        elif msg_type == 'quit':
            self.quit()
            return True
        return False

    def start_gen0(self, msg):
        assert self.child_process is None

        # TODO: once we change c++ to directly communicate game data to the training-server via TCP,
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
            '--training-server-hostname', self.training_server_host,
            '--training-server-port', self.training_server_port,
            '--starting-generation', gen,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        self_play_cmd = ' '.join(map(str, self_play_cmd))

        log_filename = os.path.join(self.organizer.logs_dir, f'self-play.log')
        with open(log_filename, 'a') as log_file:
            logger.info(f'Running gen-0 self-play: {self_play_cmd}')
            subprocess_util.run(self_play_cmd, stdout=log_file,
                                stderr=log_file, check=True)
            logger.info(f'Gen-0 self-play complete!')

    def start(self, msg):
        assert self.child_process is None

        # TODO: once we change c++ to directly communicate game data to the training-server via TCP,
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
            '--training-server-hostname', self.training_server_host,
            '--training-server-port', self.training_server_port,
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
