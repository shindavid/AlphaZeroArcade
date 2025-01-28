from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ServerStatus, Generation
from alphazero.logic.ratings import WinLossDrawCounts
from util.logging_util import get_logger
from util.socket_util import JsonDict, SocketSendException
from util import ssh_util
from util.py_util import find_largest_gap

import numpy as np
import networkx as nx

from dataclasses import dataclass
import threading
from typing import List, Optional, Set, Tuple, Dict


logger = get_logger()
N_GAMES = 100

@dataclass(frozen=True)
class Agent:
    gen: int
    n_iters: int

class BenchmarkingManager:
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller

        # W[i][j] contains the # of wins that T[i] has vs T[j], where
        #
        # T = self._tested_agents
        # W = self._W_matrix
        self._represented_gens: Set[int] = set()
        self._tested_agents: Dict[Agent, int] = {}
        self._W_matrix = np.zeros((0, 0), dtype=float)
        self.G = nx.Graph() # node is agent, edge is if they have played each other

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
            self._worker_msg_handler, conn, 'benchmarking-worker',
            disconnect_handler=self._handle_worker_disconnect)

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('benchmarking-worker received json message: %s', msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'weights-request':
            self._handle_weights_request(msg, conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('ratings-worker: unknown message type: %s', msg)
        return False

    def _handle_pause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        cond = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

    def _handle_worker_disconnect(self, conn: ClientConnection):
        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            conn.aux.pop('pending_pause_ack', None)
            conn.aux.pop('pending_unpause_ack', None)
            cond.notify_all()

        # We set the management status to DEACTIVATING, rather than INACTIVE, here, so that the
        # worker loop breaks while the server loop continues.
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.mark_as_deactivating(conn.client_domain)

    def _handle_weights_request(self, msg: JsonDict, conn: ClientConnection):
        gen = msg['gen']
        thread = threading.Thread(target=self._manage_worker, args=(gen, conn),
                                  daemon=True, name=f'manage-benchmarking-worker')
        thread.start()

    def _manage_worker(self, gen: Generation, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id

            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            self._pause(conn)
            self._update_weights(gen, conn)

            while table.active(domain):
                if not table.acquire_lock(domain):
                    break
                self._unpause(conn)
                if table.wait_for_lock_expiry(domain):
                    self._pause(conn)
                    table.release_lock(domain)
        except SocketSendException:
            logger.warning('Error sending to %s - worker likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _pause(self, conn: ClientConnection):
        logger.debug('Pausing %s...', conn)
        data = {
            'type': 'pause',
        }
        conn.aux['pending_pause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_pause_ack' not in conn.aux)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)
        data = {
            'type': 'unpause',
        }
        conn.aux['pending_unpause_ack'] = True
        conn.socket.send_json(data)

        cond: threading.Condition = conn.aux['ack_cond']
        with cond:
            cond.wait_for(lambda: 'pending_unpause_ack' not in conn.aux)

        logger.debug('Unpause of %s complete!', conn)

    def _update_weights(self, gen: Generation, conn: ClientConnection):
        self._controller.broadcast_weights(conn, gen)
        conn.aux['gen'] = gen

    def notify_of_new_model(self):
        """
        Notify manager that there is new work to do.
        """
        self._set_priority()
        with self._lock:
            self._new_work_cond.notify_all()

    # def _set_priority(self):
    #     # TODO: do something here to compute an elevate bool, and then call
    #     #
    #     # self._controller.set_ratings_priority(elevate)
    #     #
    #     # There is a question here of whether to use that function as-is, in which case we probably
    #     # need to fold the BENCHMARKING_* roles into the RATINGS domain, or whether to add a new
    #     # BENCHMARKING domain and generalize that function and pass in the domain as an argument.
    #     raise NotImplementedError

    def _set_priority(self):
        latest_gen = self._controller.latest_gen()
        dict_len = len(self._represented_gens)
        benchmarking_in_progress = False

        # The target_rate is the % of the generations that the ratings manager aims to rate. Its
        # dictate is to never allow the % of rated generations to exceed this target_rate.
        #
        # To illustrate, if there are 2 generations and the ratings manager hasn't yet done
        # anything, and we are using a target_rate of 10%, it *cannot* start rating anything,
        # because doing so risks exceeding the target_rate (since 1 / 2 = 50% > 10%).
        #
        # In order to properly respect the target_rate, then, we need to add 1 to the numerator
        # when calculating current_rate.
        #
        # If the ratings manager is currently rating something, however, then dict_len / latest_gen
        # in fact already represents what the current_rate *would become* if the ratings manager
        # computes another rating. So, in this case, we don't need to add 1 to the numerator.
        target_rate = self._controller.params.target_rating_rate
        num = dict_len + (0 if benchmarking_in_progress else 1)
        den = max(1, latest_gen)
        current_rate = num / den

        elevate = current_rate < target_rate
        logger.debug('Ratings elevate-priority:%s (latest=%s, dict_len=%s, in_progress=%s, '
                     'current=%.2f, target=%.2f)', elevate, latest_gen, dict_len,
                     benchmarking_in_progress, current_rate, target_rate)
        self._controller.set_ratings_priority(elevate)

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
        elif msg_type == 'benchmark-result':
            self._handle_benchmark_result(msg, conn)
        elif msg_type == 'model-request':
            self._handle_model_request(msg, conn)
        else:
            logger.warning('ratings-server: unknown message type: %s', msg)
        return False

    def _handle_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux['status_cond']
        with status_cond:
            conn.aux['status'] = ServerStatus.READY
            status_cond.notify_all()

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

    def _handle_model_request(self, msg: JsonDict, conn: ClientConnection):
        gen = msg['gen']
        assert gen <= self._controller.latest_gen(), (gen, self._controller.latest_gen())

        self._controller.broadcast_weights(conn, gen)

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
        n = self._controller.latest_gen()
        print(f'gen: {n}')

        k = int(np.log2(n))
        cutoffs = n - np.power(2, np.arange(k + 1))
        cutoffs = np.concatenate([cutoffs + 1, [0]])
        agents = [Agent(gen=i, n_iters=0) for i in range(0, n + 1)]
        intervals = [agents[cutoffs[i + 1] : (cutoffs[i])] for i in range(len(cutoffs)) if i!= len(cutoffs) - 1]

        commitee = []
        for i in intervals:
            commitee.append(sorted(i, key=lambda x: (self.G.degree(x) if x in self.G.nodes else 0, x.gen, -x.n_iters))[-1])

        rep = commitee[0]

        data = {
            'type': 'match-request',
            'gen1': n,
            'gen2': rep.gen,
            'n_iters1': 0,
            'n_iters2': rep.n_iters,
            'n_games': N_GAMES,
        }
        conn.socket.send_json(data)

    def _get_next_gen_to_benchmark(self) -> int:
        """
        Returns the next generation to rate. Assumes that there is at least one generation that has
        not been rated and is not currently being rated.

        Description of selection algorithm:

        Let G be the set of gens that we have graded or are currently graded thus far, and let M be
        the max generation that exists in the models directory.

        If M is at least 10 greater than the max element of G, then we return M. This is to keep up
        with a currently running alphazero run.

        Otherwise, if 1 is not in G, then we return 1.

        Finally, we find the largest gap in G, and return the midpoint of that gap. If G is fully
        saturated, we return M, which cannot be in G due to the above assumption.
        """
        latest_gen = self._controller.latest_gen()
        assert latest_gen > 0, latest_gen

        logger.debug('Getting next gen to rate, latest_gen=%s...', latest_gen)
        with self._lock:
            taken_gens = [g for g, r in self._rating_data_dict.items()
                          if r.rating is not None or r.owner is not None]
            taken_gens.sort()
        if not taken_gens:
            logger.debug('No gens yet rated, rating latest (%s)...', latest_gen)
            return latest_gen

        max_taken_gen = taken_gens[-1]

        assert latest_gen >= max_taken_gen
        latest_gap = latest_gen - max_taken_gen

        latest_gap_threshold = 10
        if latest_gap >= latest_gap_threshold:
            logger.debug('%s+ gap to latest, rating latest (%s)...', latest_gap_threshold,
                         latest_gen)
            return latest_gen

        if taken_gens[0] > 1:
            logger.debug('Gen-1 not yet rated, rating it...')
            return 1

        assert latest_gen != 1, latest_gen

        if len(taken_gens) == 1:
            logger.debug('No existing gaps, rating latest (%s)...', latest_gen)
            return latest_gen

        left, right = find_largest_gap(taken_gens)
        gap = right - left
        if 2 * latest_gap >= gap:
            logger.debug(
                'Large gap to latest, rating latest=%s '
                '(biggest-gap:[%s, %s], latest-gap:[%s, %s])...',
                latest_gen, left, right, max_taken_gen, latest_gap)
            return latest_gen

        assert max(gap, latest_gap) > 1, (gap, latest_gap)

        if left + 1 == right:
            assert latest_gen > right, (latest_gen, right)
            logger.debug('No existing gaps, rating latest (%s)...', latest_gen)
            return latest_gen

        mid = (left + right) // 2
        logger.debug('Rating gen %s (biggest-gap:[%s, %s], latest=%s)...',
                     mid, left, right, latest_gen)
        return mid

    def _handle_benchmark_result(self, msg: JsonDict, conn: ClientConnection):
        gen1 = msg['gen1']
        gen2 = msg['gen2']
        n_iters1 = msg['n_iters1']
        n_iters2 = msg['n_iters2']
        agent1 = Agent(gen=gen1, n_iters=n_iters1)
        agent2 = Agent(gen=gen2, n_iters=n_iters2)
        record = WinLossDrawCounts.from_json(msg['record'])

        with self._lock:
            if agent1 not in self.G.nodes:
                ix1 = len(self.G.nodes)
                self.G.add_node(agent1, ix=ix1)
                self._expand_matrix(agent1)
            else:
                ix1 = self.G.nodes[agent1]['ix']

            if agent2 not in self.G.nodes:
                ix2 = len(self.G.nodes)
                self.G.add_node(agent2, ix=ix2)
                self._expand_matrix(agent2)
            else:
                ix2 = self.G.nodes[agent2]['ix']

            if not self.G.has_edge(agent1, agent2):
                self.G.add_edge(agent1, agent2)

            self._W_matrix[ix1, ix2] += record.win + 0.5 * record.draw
            self._W_matrix[ix2, ix1] += record.loss + 0.5 * record.draw

            print(self._W_matrix)

    def _expand_matrix(self, agent: Agent):
        n = self._W_matrix.shape[0]
        new_matrix = np.zeros((n + 1, n + 1), dtype=float)
        new_matrix[:n, :n] = self._W_matrix
        self._W_matrix = new_matrix