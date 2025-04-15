from alphazero.logic.agent_types import MCTSAgent
from alphazero.logic.custom_types import ClientRole, FileToTransfer
from alphazero.logic.ratings import extract_match_record
from alphazero.servers.gaming.server_base import ServerBase, ServerParams
from util.socket_util import JsonDict
from util.str_util import make_args_str
from util import subprocess_util

from dataclasses import dataclass
import logging
import os


logger = logging.getLogger(__name__)


@dataclass
class RatingsServerParams(ServerParams):
    rating_tag: str = ''

    @staticmethod
    def add_additional_args(group):
        defaults = RatingsServerParams()

        group.add_argument('-r', '--rating-tag', default=defaults.rating_tag,
                           help='ratings tag. Loop controller collates ratings by this str. It is '
                           'the responsibility of the user to make sure that the same '
                           'binary/params are used across different RatingsServer processes '
                           'sharing the same rating-tag. (default: "%(default)s")')


class RatingsServer(ServerBase):
    def _send_handshake(self):
        additional_data = {'rating_tag': self._params.rating_tag}
        self._session_data.send_handshake(ClientRole.RATINGS_SERVER,
                                          addtional_data=additional_data)

    def _run_match_helper(self, msg: JsonDict):
        assert not self._running
        self._running = True

        files_required = [FileToTransfer(**f) for f in msg['files_required']]
        self._session_data.request_files(files_required)

        mcts_agent = MCTSAgent(**msg['mcts_agent'])
        mcts_gen = mcts_agent.gen
        ref_strength = msg['ref_strength']
        n_games = msg['n_games']

        ps1 = mcts_agent.make_player_str(self._session_data.run_dir,
                                         args=self._session_data.game_spec.rating_player_options)
        ps2 = self._get_reference_player_str(ref_strength)

        binary = os.path.join(self._session_data.run_dir, mcts_agent.binary)
        log_filename = self._session_data.get_log_filename(self._config.worker_name)
        append_mode = not self._session_data.start_log_sync(log_filename)

        args = {
            '-G': n_games,
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': ClientRole.RATINGS_WORKER.value,
            '--manager-id': self._session_data.client_id,
            '--ratings-tag': f'"{self._params.rating_tag}"',
            '--cuda-device': self._params.cuda_device,
            '--do-not-report-metrics': None,
            '--log-filename': log_filename,
        }
        if append_mode:
            args['--log-append-mode'] = None
        args.update(self._session_data.game_spec.rating_options)
        cmd = [
            binary,
            '--player', f'"{ps1}"',
            '--player', f'"{ps2}"',
            ]
        cmd.append(make_args_str(args))
        cmd = ' '.join(map(str, cmd))

        ref_name = RatingsServer._get_reference_player_name(ref_strength)

        cwd = self._session_data.run_dir
        self._proc = subprocess_util.Popen(cmd, cwd=cwd)
        logger.info('Running %s vs %s match [%s] from %s: %s', f'MCTS-{mcts_agent.gen}', ref_name, self._proc.pid,
                    cwd, cmd)
        stdout = self._session_data.wait_for(self._proc)
        self._proc = None

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info('Match result: %s', record.get(0))

        self._running = False

        data = {
            'type': 'match-result',
            'record': record.get(0).to_json(),
            'mcts_gen': mcts_gen,
            'ref_strength': ref_strength,
        }

        self._session_data.socket.send_json(data)
        self._send_ready()


    @staticmethod
    def _get_reference_player_name(strength: int):
        return f'ref-{strength}'

    def _get_reference_player_str(self, strength: int):
        name = RatingsServer._get_reference_player_name(strength)
        family = self._session_data.game_spec.reference_player_family
        type_str = family.type_str
        strength_param = family.strength_param

        player_args = {
            '--type': type_str,
            '--name': name,
            strength_param: strength,
        }
        return make_args_str(player_args)
