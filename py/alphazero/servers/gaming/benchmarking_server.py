from alphazero.logic.build_params import BuildParams
from alphazero.logic.custom_types import ClientRole
from alphazero.logic.ratings import extract_match_record
from alphazero.logic.shutdown_manager import ShutdownManager
from alphazero.logic.signaling import register_standard_server_signals
from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.session_data import SessionData
from util.logging_util import LoggingParams, get_logger
from util.socket_util import JsonDict, SocketRecvException, SocketSendException
from util.str_util import make_args_str
from util import subprocess_util

from dataclasses import dataclass, fields
import threading


logger = get_logger()


@dataclass
class BenchmarkingServerParams(BaseParams):
    @staticmethod
    def create(args) -> 'BenchmarkingServerParams':
        kwargs = {f.name: getattr(args, f.name) for f in fields(BenchmarkingServerParams)}
        return BenchmarkingServerParams(**kwargs)

    @staticmethod
    def add_args(parser, omit_base=False):
        group = parser.add_argument_group(f'BenchmarkingServer options')
        if not omit_base:
            BaseParams.add_base_args(group)


class BenchmarkingServer:
    def __init__(self, params: BenchmarkingServerParams, logging_params: LoggingParams,
                 build_params: BuildParams):
        self._params = params
        self._build_params = build_params
        self._session_data = SessionData(params, logging_params)
        self._shutdown_manager = ShutdownManager()
        self._running = False
        self._worker_log_sync_registered = False

        register_standard_server_signals(ignore_sigint=params.ignore_sigint)

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
        self._session_data.send_handshake(ClientRole.BENCHMARKING_SERVER)

    def _recv_handshake(self):
        self._session_data.recv_handshake(ClientRole.BENCHMARKING_SERVER)

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

    def _send_ready(self):
        data = { 'type': 'ready', }
        self._session_data.socket.send_json(data)

    def _handle_msg(self, msg: JsonDict) -> bool:
        logger.debug('self-ratings-server received json message: %s', msg)

        msg_type = msg['type']
        if msg_type == 'match-request':
            self._handle_match_request(msg)
        elif msg_type == 'quit':
            self._quit()
            return True
        else:
            raise Exception('Unknown message type: %s', msg_type)
        return False

    def _handle_match_request(self, msg: JsonDict):
        thread = threading.Thread(target=self._run_match, args=(msg,), daemon=True,
                                  name='run-match')
        thread.start()

    def _quit(self):
        logger.info('Received quit command')
        self._shutdown_manager.request_shutdown(0)

    def _run_match(self, msg: JsonDict):
        try:
            self._run_match_helper(msg)
        except:
            logger.error('Unexpected error in run-match:', exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    def _run_match_helper(self, msg: JsonDict):
        assert not self._running
        self._running = True

        # gen==0 means no model
        gen1 = msg['gen1']
        n_iters1 = msg['n_iters1']
        gen2 = msg['gen2']
        n_iters2 = msg['n_iters2']
        n_games = msg['n_games']

        ps1 = self._get_mcts_player_str(gen1, n_iters1)
        ps2 = self._get_mcts_player_str(gen2, n_iters2)
        binary = self._build_params.get_binary_path(self._session_data.game)

        log_filename1 = self._session_data.get_log_filename('self-ratings-worker', 'a')
        log_filename2 = self._session_data.get_log_filename('self-ratings-worker', 'b')
        if not self._worker_log_sync_registered:
            self._session_data.start_log_sync(log_filename1)
            self._session_data.start_log_sync(log_filename2)
            self._worker_log_sync_registered = True

        # TODO: We launch 2 workers from this single server. They will share the same manager-id.
        # There is currently a check on the loop-controller side that will cause the second
        # worker connection to be rejected. We should fix this.
        #
        # (dshin) I *believe* that there were 2 reasons that the loop-controller cares about
        # manager-id's:
        #
        # 1. To catch code errors leading to a server mistakenly spawning multiple workers
        #
        # 2. To ensure all ratings-workers spawned by a single ratings server log to the same log
        #    file.
        #
        # Reason 2 became outdated with the merge of the local-logging branch on 2024-01-20.
        # Reason 1 is probably not too relevant now that this code has become more mature, but if
        # we want to keep it, we can perhaps change it to allow up to 2 workers, either in
        # general, or for specific client-roles.
        base_args = {
            '-G': n_games,
            '--loop-controller-hostname': self._params.loop_controller_host,
            '--loop-controller-port': self._params.loop_controller_port,
            '--client-role': ClientRole.BENCHMARKING_WORKER.value,
            '--manager-id': self._session_data.client_id,  # see comment above
            '--cuda-device': self._params.cuda_device,
            '--do-not-report-metrics': None,
        }

        args1 = dict(base_args)
        args1.update({
            '--weights-request-generation': gen1,
            '--log-filename': log_filename1,
        })
        args2 = dict(base_args)
        args2.update({
            '--weights-request-generation': gen2,
            '--log-filename': log_filename2,
        })

        port = 1234  # TODO: move this to constants.py or somewhere

        cmd1 = [
            binary,
            '--port', str(port),
            '--player', f'"{ps1}"',
        ]
        cmd1.append(make_args_str(args1))
        cmd1 = ' '.join(map(str, cmd1))

        cmd2 = [
            binary,
            '--remote-port', str(port),
            '--player', f'"{ps2}"',
        ]
        cmd2.append(make_args_str(args2))
        cmd2 = ' '.join(map(str, cmd2))

        p1 = BenchmarkingServer._get_mcts_player_name(gen1, n_iters1)
        p2 = BenchmarkingServer._get_mcts_player_name(gen2, n_iters2)

        proc1 = subprocess_util.Popen(cmd1)
        proc2 = subprocess_util.Popen(cmd2)
        logger.info('Running %s vs %s match [%s] [%s]', p1, p2, proc1.pid, proc2.pid)
        logger.info('Command 1: %s', cmd1)
        logger.info('Command 2: %s', cmd2)

        stdout = self._session_data.wait_for(proc1)

        # NOTE: extracting the match record from stdout is potentially fragile. Consider
        # changing this to have the c++ process directly communicate its win/loss data to the
        # loop-controller. Doing so would better match how the self-play server works.
        record = extract_match_record(stdout)
        logger.info('Match result: %s', record.get(0))

        self._running = False

        data = {
            'type': 'match-result',
            'record': record.get(0).to_json(),
            'gen1': gen1,
            'gen2': gen2,
            'n_iters1': n_iters1,
            'n_iters2': n_iters2,
        }

        self._session_data.socket.send_json(data)
        self._send_ready()

    @staticmethod
    def _get_mcts_player_name(gen: int, n_iters: int):
        return f'MCTS-{gen}-{n_iters}'

    def _get_mcts_player_str(self, gen: int, n_iters: int):
        name = BenchmarkingServer._get_mcts_player_name(gen, n_iters)

        player_args = {
            '--type': 'MCTS-C',
            '--name': name,
            '--cuda-device': self._params.cuda_device,
            '-i': n_iters,
        }
        if gen == 0:
            player_args['--no-model'] = None

        return make_args_str(player_args)
