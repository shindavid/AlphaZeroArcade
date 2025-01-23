from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface


from alphazero.logic.custom_types import ClientConnection, ServerStatus
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util

import numpy as np

from dataclasses import dataclass
import threading
from typing import List, Optional, Set


logger = get_logger()
N_GAMES = 100

@dataclass
class Agent:
    gen: int
    n_iters: int
    cur_rating: Optional[float] = None


class BenchmarkingManager:
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller

        # W[i][j] contains the # of wins that T[i] has vs T[j], where
        #
        # T = self._tested_agents
        # W = self._W_matrix
        self._represented_gens: Set[int] = set()
        self._tested_agents: List[Agent] = []
        self._W_matrix = np.zeros((0, 0), dtype=float)

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cond = threading.Condition(self._lock)

    def add_server(self, conn: ClientConnection):
        ssh_pub_key = ssh_util.get_pub_key()
        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
            'game': self._controller.game_spec.name,
            'tag': self._controller.run_params.tag,
            'ssh_pub_key': ssh_pub_key,
        }
        conn.socket.send_json(reply)

        conn.aux['status_cond'] = threading.Condition()
        conn.aux['status'] = ServerStatus.BLOCKED

        self._start()
        logger.info('Starting benchmarking-recv-loop for %s...', conn)
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, 'benchmarking-server',
            disconnect_handler=self._handle_server_disconnect)

        thread = threading.Thread(target=self._manage_server, args=(conn,),
                                  daemon=True, name=f'manage-benchmarking-server')
        thread.start()

    def add_worker(self, conn: ClientConnection):
        conn.aux['ack_cond'] = threading.Condition()

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, 'ratings-worker',
            disconnect_handler=self._handle_worker_disconnect)

    def notify_of_new_model(self):
        """
        Notify manager that there is new work to do.
        """
        self._set_priority()
        with self._lock:
            self._new_work_cond.notify_all()

    def _set_priority(self):
        # TODO: do something here to compute an elevate bool, and then call
        #
        # self._controller.set_ratings_priority(elevate)
        #
        # There is a question here of whether to use that function as-is, in which case we probably
        # need to fold the BENCHMARKING_* roles into the RATINGS domain, or whether to add a new
        # BENCHMARKING domain and generalize that function and pass in the domain as an argument.
        raise NotImplementedError

    def _start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            # self._load_past_data()

    # def _load_past_data(self):
    #     logger.info('Loading past ratings data...')
    #     conn = self._controller.ratings_db_conn_pool.get_connection()
    #     c = conn.cursor()
    #     res = c.execute('SELECT mcts_gen, ref_strength, mcts_wins, draws, ref_wins FROM matches WHERE tag = ?',
    #                     (self._tag,))

    #     for mcts_gen, ref_strength, mcts_wins, draws, ref_wins in res.fetchall():
    #         if mcts_gen not in self._rating_data_dict:
    #             data = RatingData(mcts_gen, self._min_ref_strength, self._max_ref_strength)
    #             self._rating_data_dict[mcts_gen] = data
    #         counts = WinLossDrawCounts(mcts_wins, ref_wins, draws)
    #         self._rating_data_dict[mcts_gen].add_result(ref_strength, counts, set_rating=False)

    #     for data in self._rating_data_dict.values():
    #         data.set_rating()

    #     for gen, data in self._rating_data_dict.items():
    #         if data.rating is None:
    #             data.est_rating = self._estimate_rating(gen)

    #     self._set_priority()

    def _server_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('ratings-server received json message: %s', msg)

        if msg_type == 'ready':
            self._handle_ready(conn)
        elif msg_type == 'log-sync-start':
            self._controller.start_log_sync(conn, msg['log_filename'])
        elif msg_type == 'log-sync-stop':
            self._controller.stop_log_sync(conn, msg['log_filename'])
        elif msg_type == 'match-result':
            self._handle_match_result(msg, conn)
        else:
            logger.warning('ratings-server: unknown message type: %s', msg)
        return False

    def _handle_server_disconnect(self, conn: ClientConnection):
        gen = conn.aux.pop('gen', None)
        if gen is not None:
            with self._lock:
                rating_data = self._rating_data_dict.get(gen, None)
                if rating_data is not None:
                    rating_data.owner = None

        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.deactivate(conn.client_domain)

        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.DISCONNECTED
            status_cond.notify_all()

    def _manage_server(self, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id
            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            table.activate(domain)

            # NOTE: the worker loop breaks when the table becomes DEACTIVATING, while this loop
            # only breaks when the table becomes INACTIVE. It is important then to use
            # (not inactive) in the below loop-condition, rather than (active).
            while not table.inactive(domain):
                status = self._wait_for_unblock(conn)
                if status == ServerStatus.DISCONNECTED:
                    break
                if conn.aux.get('gen', None) is None:
                    self._wait_until_work_exists()

                table.activate(domain)
                if not table.acquire_lock(domain):
                    break
                self._send_match_request(conn)

                # We do not release the lock here. The lock is released either when a gen is
                # fully rated, or when the server disconnects.
        except SocketSendException:
            logger.warning('Error sending to %s - server likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _handle_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.READY
            status_cond.notify_all()

    def _wait_for_unblock(self, conn: ClientConnection) -> ServerStatus:
        """
        The server status is initially BLOCKED. This function waits until that status is
        changed (either to READY or DISCONNECTED). After waiting, it resets the status to
        BLOCKED, and returns what the status was changed to.
        """
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            status_cond.wait_for(lambda: conn.aux['status'] != ServerStatus.BLOCKED)
            status = conn.aux['status']
            conn.aux['status'] = ServerStatus.BLOCKED
            return status

    def _wait_until_work_exists(self):
        with self._lock:
            self._new_work_cond.wait_for(
                lambda: len(self._represented_gens) < self._controller.latest_gen())

    def _send_match_request(self, conn: ClientConnection):
        gen = conn.aux.get('gen', None)
        if gen is None:
            gen = self._get_next_gen_to_rate()
            conn.aux['gen'] = gen

        gen1, gen2 = self._get_next_gen_to_benchmark()
        rating_data = self._get_rating_data(conn, gen)
        assert rating_data.rating is None
        strength = rating_data.get_next_strength_to_test()
        assert strength is not None

        # Agent? here
        data = {
            'type': 'match-request',
            'gen1': gen1,
            'gen2': gen2,
            'ref_strength': strength,
            'n_games': N_GAMES,
        }
        conn.socket.send_json(data)

    def _get_next_gen_to_benchmark(self) -> int, int:
        # TODO: pick two gens to benchmark
        raise NotImplementedError

