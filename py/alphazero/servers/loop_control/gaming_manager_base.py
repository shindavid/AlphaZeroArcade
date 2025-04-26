from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import ClientConnection, Domain, ServerStatus
from util.socket_util import JsonDict, SocketSendException

from abc import abstractmethod
from dataclasses import dataclass, field
import logging
import threading
from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


@dataclass
class ServerAuxBase:
    """
    Auxiliary data stored per server connection.
    """
    status_cond: threading.Condition = field(default_factory=threading.Condition)
    status: ServerStatus = ServerStatus.BLOCKED

    @abstractmethod
    def work_in_progress(self) -> bool:
        """
        Returns True if the server is in progress of work.
        """
        pass


@dataclass
class WorkerAux:
    """
    Auxiliary data stored per worker connection.
    """
    cond: threading.Condition = field(default_factory=threading.Condition)
    pending_pause_ack: bool = False
    pending_unpause_ack: bool = False


@dataclass
class ManagerConfig:
    server_aux_class: Type
    worker_aux_class: Type
    server_name: str
    worker_name: str
    domain: Domain


class GamingManagerBase:
    def __init__(self, controller: LoopController, manager_config: ManagerConfig, tag: str=None):
        self._tag = tag
        self._controller: LoopController = controller
        self._config: ManagerConfig = manager_config

        self._started = False
        self._lock = threading.Lock()
        self._new_work_cond = threading.Condition(self._lock)

    def add_server(self, conn: ClientConnection):
        conn.aux = self._config.server_aux_class()
        self._controller.send_handshake_ack(conn)

        self._start()
        logger.info('Starting %s recv-loop for %s...', self._config.server_name, conn)
        self._controller.launch_recv_loop(
            self._server_msg_handler, conn, self._config.server_name,
            disconnect_handler=self.handle_server_disconnect)

        thread = threading.Thread(target=self._manage_server, args=(conn,),
                                  daemon=True, name=f'manage-{self._config.server_name}')
        thread.start()

    def add_worker(self, conn: ClientConnection):
        conn.aux = self._config.worker_aux_class()

        reply = {
            'type': 'handshake-ack',
            'client_id': conn.client_id,
        }
        conn.socket.send_json(reply)
        self._controller.launch_recv_loop(
            self._worker_msg_handler, conn, self._config.worker_name,
            disconnect_handler=self._handle_worker_disconnect)

    def notify_of_new_model(self):
        """
        Notify manager that there is new work to do.
        """
        self.set_priority()
        with self._lock:
            self._new_work_cond.notify_all()

    def _start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self.load_past_data()

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
                if not conn.aux.work_in_progress():
                    logger.debug(f'waiting for work to do')
                    self._wait_until_work_exists()

                logger.debug(f"Managing {self._config.server_name}, priority: {table}")
                table.activate(domain)
                if not table.acquire_lock(domain):
                    break
                self.send_match_request(conn)

                # We do not release the lock here. The lock is released either when a gen is
                # fully rated, or when the server disconnects.
        except SocketSendException:
            logger.warning('Error sending to %s - server likely disconnected', conn)
        except:
            logger.error('Unexpected error managing %s', conn, exc_info=True)
            self._controller.request_shutdown(1)

    def _wait_until_work_exists(self):
        with self._lock:
            self._new_work_cond.wait_for(self._has_work)

    def _has_work(self) -> bool:
        logger.debug(f'num_evaluated_gens={self.num_evaluated_gens()}, latest_gen={self._controller.latest_gen()}')
        return self.num_evaluated_gens() < self._controller.latest_gen()

    def _wait_for_unblock(self, conn: ClientConnection) -> ServerStatus:
        """
        The server status is initially BLOCKED. This function waits until that status is
        changed (either to READY or DISCONNECTED). After waiting, it resets the status to
        BLOCKED, and returns what the status was changed to.
        """
        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            status_cond.wait_for(lambda: conn.aux.status != ServerStatus.BLOCKED)
            status = conn.aux.status
            conn.aux.status = ServerStatus.BLOCKED
            return status

    def _server_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('%s received json message: %s',
                     self._config.server_name, msg)

        if msg_type == 'ready':
            self._set_ready(conn)
        elif msg_type == 'log-sync-start':
            self._controller.start_log_sync(conn, msg['log_filename'])
        elif msg_type == 'log-sync-stop':
            self._controller.stop_log_sync(conn, msg['log_filename'])
        elif msg_type == 'match-result':
            self.handle_match_result(msg, conn)
        elif msg_type == 'file-request':
            self._handle_file_request(conn, msg['files'])
        else:
            logger.warning('%s: unknown message type: %s',
                           self._config.server_name, msg)
        return False

    def _set_ready(self, conn: ClientConnection):
        status_cond: threading.Condition = conn.aux.status_cond
        with status_cond:
            conn.aux.status = ServerStatus.READY
            status_cond.notify_all()

    def _set_domain_priority(self, dict_len: int, rating_in_progress: bool):
        latest_gen = self._controller.organizer.get_latest_model_generation(default=0)
        target_rate = self._controller.params.target_rating_rate
        num = dict_len + (0 if rating_in_progress else 1)
        den = max(1, latest_gen)
        current_rate = num / den

        elevate = current_rate < target_rate
        logger.debug('%s elevate-priority:%s (latest=%s, dict_len=%s, in_progress=%s, '
                     'current=%.2f, target=%.2f)', self._config.domain.value, elevate, latest_gen, dict_len,
                     rating_in_progress, current_rate, target_rate)
        self._controller.set_domain_priority(self._config.domain, elevate)

    def _worker_msg_handler(self, conn: ClientConnection, msg: JsonDict) -> bool:
        msg_type = msg['type']
        logger.debug('%s received json message: %s', self._config.worker_name, msg)

        if msg_type == 'pause-ack':
            self._handle_pause_ack(conn)
        elif msg_type == 'unpause-ack':
            self._handle_unpause_ack(conn)
        elif msg_type == 'worker-ready':
            self._handle_worker_ready(conn)
        elif msg_type == 'done':
            return True
        else:
            logger.warning('%s: unknown message type: %s', self._config.worker_name, msg)
        return False

    def _handle_worker_ready(self, conn: ClientConnection):
        thread = threading.Thread(target=self._manage_worker, args=(conn,),
                                  daemon=True,
                                  name=f'manage-{self._config.worker_name}')
        thread.start()

    def _manage_worker(self, conn: ClientConnection):
        try:
            domain = conn.client_domain
            gpu_id = conn.client_gpu_id

            table: GpuContentionTable = self._controller.get_gpu_lock_table(gpu_id)
            self._pause(conn)

            logger.debug(f"{domain} active: {table.active(domain)}")
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

    def _handle_worker_disconnect(self, conn: ClientConnection):
        aux: WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

        # We set the management status to DEACTIVATING, rather than INACTIVE, here, so that the
        # worker loop breaks while the server loop continues.
        table: GpuContentionTable = self._controller.get_gpu_lock_table(conn.client_gpu_id)
        table.mark_as_deactivating(conn.client_domain)

    def _pause(self, conn: ClientConnection):
        logger.debug('Pausing %s...', conn)

        aux: WorkerAux = conn.aux
        aux.pending_pause_ack = True

        conn.socket.send_json({ 'type': 'pause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_pause_ack)

        logger.debug('Pause of %s complete!', conn)

    def _unpause(self, conn: ClientConnection):
        logger.debug('Unpausing %s...', conn)

        aux: WorkerAux = conn.aux
        aux.pending_unpause_ack = True

        conn.socket.send_json({ 'type': 'unpause' })

        with aux.cond:
            aux.cond.wait_for(lambda: not aux.pending_unpause_ack)

        logger.debug('Unpause of %s complete!', conn)

    def _handle_pause_ack(self, conn: ClientConnection):
        aux: WorkerAux = conn.aux
        with aux.cond:
            aux.pending_pause_ack = False
            aux.cond.notify_all()

    def _handle_unpause_ack(self, conn: ClientConnection):
        aux: WorkerAux = conn.aux
        with aux.cond:
            aux.pending_unpause_ack = False
            aux.cond.notify_all()

    def _handle_file_request(self, conn: ClientConnection, files: List[JsonDict]):
        self._controller.handle_file_request(conn, files)

    @abstractmethod
    def set_priority(self):
        pass

    @abstractmethod
    def load_past_data(self):
        pass

    @abstractmethod
    def num_evaluated_gens(self) -> int:
        pass

    @abstractmethod
    def handle_server_disconnect(self, conn: ClientConnection):
        pass

    @abstractmethod
    def send_match_request(self, conn: ClientConnection):
        pass

    @abstractmethod
    def handle_match_result(self, msg: JsonDict, conn: ClientConnection):
        pass
