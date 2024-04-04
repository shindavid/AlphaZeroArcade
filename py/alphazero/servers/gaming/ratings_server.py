from alphazero.logic.custom_types import ClientRole
from alphazero.logic.ratings import extract_match_record
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
class RatingsServerParams(BaseParams):
    n_search_threads: int = 4
    parallelism_factor: int = 100

    @staticmethod
    def create(args) -> 'RatingsServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(RatingsServerParams)}
        return RatingsServerParams(**kwargs)

    @staticmethod
    def add_args(parser, omit_base=False):
        defaults = RatingsServerParams()

        group = parser.add_argument_group(f'RatingsServer options')
        if not omit_base:
            BaseParams.add_base_args(group)

        group.add_argument('-n', '--n-search-threads', type=int, default=defaults.n_search_threads,
                           help='num search threads per game (default: %(default)s)')
        group.add_argument('-p', '--parallelism-factor', type=int,
                           default=defaults.parallelism_factor,
                           help='parallelism factor (default: %(default)s)')


class RatingsServer(GameServerBase):
    def __init__(self, params: RatingsServerParams, logging_params: LoggingParams):
        super().__init__(params, logging_params, ClientRole.RATINGS_SERVER)
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

    def send_ready(self):
        data = {
            'type': 'ready',
            }
        self.loop_controller_socket.send_json(data)

    def recv_loop_prelude(self):
        self.send_ready()

    def handle_msg(self, msg: JsonDict) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'ratings-server received json message: {msg}')

        msg_type = msg['type']
        if msg_type == 'match-request':
            self._handle_match_request(msg)
        elif msg_type == 'quit':
            self.quit()
            return True
        else:
            raise Exception(f'Unknown message type: {msg_type}')
        return False

    def _handle_match_request(self, msg: JsonDict):
        thread = threading.Thread(target=self._run_match, args=(msg,),
                                  daemon=True, name=f'run-match')
        thread.start()

    def _run_match(self, msg: JsonDict):
        try:
            self._run_match_helper(msg)
        except:
            logger.error(f'Unexpected error in run-match:', exc_info=True)
            self.shutdown_manager.request_shutdown(1)

    def _run_match_helper(self, msg: JsonDict):
        assert not self._running
        self._running = True

        mcts_gen = msg['mcts_gen']
        ref_strength = msg['ref_strength']
        n_games = msg['n_games']
        n_mcts_iters = msg['n_mcts_iters']

        n_search_threads = self.params.n_search_threads
        parallelism_factor = self.params.parallelism_factor

        ps1 = self.get_mcts_player_str(mcts_gen, n_mcts_iters, n_search_threads)
        ps2 = self.get_reference_player_str(ref_strength)
        binary = self.binary_path
        cmd = [
            binary,
            '-G', n_games,
            '--loop-controller-hostname', self.loop_controller_host,
            '--loop-controller-port', self.loop_controller_port,
            '--client-role', ClientRole.RATINGS_WORKER.value,
            '--cuda-device', self.cuda_device,
            '--weights-request-generation', mcts_gen,
            '--do-not-report-metrics',
            '-p', parallelism_factor,
            '--player', f'"{ps1}"',
            '--player', f'"{ps2}"',
            ]
        cmd = ' '.join(map(str, cmd))

        mcts_name = RatingsServer.get_mcts_player_name(mcts_gen)
        ref_name = RatingsServer.get_reference_player_name(ref_strength)

        proc = subprocess_util.Popen(cmd)
        logger.info(f'Running {mcts_name} vs {ref_name} match [{proc.pid}]: {cmd}')
        stdout_buffer = []
        self.forward_output('ratings-worker', proc, stdout_buffer, close_remote_log=False)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout_buffer)
        logger.info(f'Match result: {record.get(0)}')

        self._running = False

        data = {
            'type': 'match-result',
            'record': record.get(0).to_json(),
            'mcts_gen': mcts_gen,
            'ref_strength': ref_strength,
        }

        self.loop_controller_socket.send_json(data)
        self.send_ready()

    @staticmethod
    def get_mcts_player_name(gen: int):
        return f'MCTS-{gen}'

    def get_mcts_player_str(self, gen: int, n_mcts_iters: int, n_search_threads: int):
        name = RatingsServer.get_mcts_player_name(gen)

        player_args = [
            '--type=MCTS-C',
            '--name', name,
            '-i', n_mcts_iters,
            '-n', n_search_threads,
            '--cuda-device', self.cuda_device,
        ]

        return ' '.join(map(str, player_args))

    @staticmethod
    def get_reference_player_name(strength: int):
        return f'ref-{strength}'

    def get_reference_player_str(self, strength: int):
        name = RatingsServer.get_reference_player_name(strength)
        family = self.game_spec.reference_player_family
        type_str = family.type_str
        strength_param = family.strength_param

        player_args = [
            '--type', type_str,
            '--name', name,
            strength_param, strength,
        ]

        return ' '.join(map(str, player_args))
