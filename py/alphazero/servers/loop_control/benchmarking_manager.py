from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ServerStatus
from util.logging_util import get_logger
from util import ssh_util

import numpy as np

from dataclasses import dataclass
import threading
from typing import List, Optional, Set


logger = get_logger()


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
