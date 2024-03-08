from alphazero.logic.common_params import CommonParams
from alphazero.logic.custom_types import ClientType
from alphazero.logic.game_server_base import GameServerBase, GameServerBaseParams
from alphazero.logic.ratings import extract_match_record
from util.logging_util import get_logger
from util.socket_util import JsonDict, send_json
from util import subprocess_util

from dataclasses import dataclass
import subprocess
import threading


logger = get_logger()


N_GAMES = 100


@dataclass
class RatingsServerParams(GameServerBaseParams):
    n_search_threads: int = 4
    parallelism_factor: int = 100

    @staticmethod
    def add_args(parser):
        defaults = RatingsServerParams()

        group = GameServerBaseParams.add_args_helper(parser, 'RatingsServer')

        group.add_argument('-n', '--n-search-threads', type=int, default=defaults.n_search_threads,
                           help='num search threads per game (default: %(default)s)')
        group.add_argument('-p', '--parallelism-factor', type=int,
                           default=defaults.parallelism_factor,
                           help='parallelism factor (default: %(default)s)')


class RatingsServer(GameServerBase):
    def __init__(self, params: RatingsServerParams, common_params: CommonParams):
        super().__init__(params, common_params, ClientType.RATINGS_MANAGER)
        self.params = params

    def request_work(self):
        data = {
            'type': 'work-request',
            }
        send_json(self.loop_controller_socket, data)

    def recv_loop_prelude(self):
        self.request_work()

    def handle_msg(self, msg: JsonDict) -> bool:
        msg_type = msg['type']
        if msg_type == 'match-request':
            self.run_func_in_new_thread(self.handle_match_request, args=(msg,))
        elif msg_type == 'quit':
            self.quit()
            return True
        return False

    def handle_match_request(self, msg: JsonDict):
        assert self.child_process is None
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
            '--client-role', ClientType.RATINGS_WORKER.value,
            '-p', parallelism_factor,
            '--player', f'"{ps1}"',
            '--player', f'"{ps2}"',
            ]
        cmd = ' '.join(map(str, cmd))

        mcts_name = RatingsServer.get_mcts_player_name(mcts_gen)
        ref_name = RatingsServer.get_reference_player_name(ref_strength)

        proc = subprocess_util.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.child_process = proc
        logger.info(f'Running {mcts_name} vs {ref_name} match: {cmd}')

        stdout, _ = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f'proc exited with code {proc.returncode}')

        # NOTE: extracting the match record from the stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info(f'Match result: {record.get(0)}')

        data = {
            'type': 'match-result',
            'record': record.get(0).to_json(),
            'mcts_gen': mcts_gen,
            'ref_strength': ref_strength,
        }

        send_json(self.loop_controller_socket, data)
        self.child_process = None
        self.request_work()

    @staticmethod
    def get_mcts_player_name(gen: int):
        return f'MCTS-{gen}'

    def get_mcts_player_str(self, gen: int, n_mcts_iters: int, n_search_threads: int):
        name = RatingsServer.get_mcts_player_name(gen)
        model = self.organizer.get_model_filename(gen)
        return f'--type=MCTS-C --name={name} -i {n_mcts_iters} -m {model} -n {n_search_threads}'

    @staticmethod
    def get_reference_player_name(strength: int):
        return f'ref-{strength}'

    def get_reference_player_str(self, strength: int):
        name = RatingsServer.get_reference_player_name(strength)
        family = self.game_spec.reference_player_family
        type_str = family.type_str
        strength_param = family.strength_param
        return f'--type={type_str} --name={name} {strength_param}={strength}'
