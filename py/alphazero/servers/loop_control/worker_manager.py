from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientId, ClientRole, Generation, GpuInfo
from util.logging_util import get_logger
from util.socket_util import send_json, send_file, SocketSendException

import threading
from typing import List, Set


logger = get_logger()


class WorkerManager:
    """
    There are two types of external worker clients: SELF_PLAY_WORKER and RATINGS_WORKER. Generally,
    the interactions with these workers are handled by SelfPlayManager and RatingsManager,
    respectively. However, there are some interactions that are common to both types of workers.
    This class handles those common interactions.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._lock = threading.Lock()
        self._pauses_acked_cv = threading.Condition(self._lock)
        self._pause_set: Set[ClientId] = set()

    def handle_disconnect(self, conn: ClientConnection):
        with self._lock:
            self._pause_set.discard(conn.client_id)

    def handle_pause_ack(self, conn: ClientConnection):
        with self._lock:
            self._pause_set.discard(conn.client_id)
            if not self._pause_set:
                self._pauses_acked_cv.notify_all()

    def pause(self, gpu_info: GpuInfo):
        """
        Issues a pause command to all workers registered with the given gpu_info. Wait for the
        workers to acknowledge the pause command before returning.
        """
        self_play_workers = self._controller.get_connections(ClientRole.SELF_PLAY_WORKER, gpu_info)
        ratings_workers = self._controller.get_connections(ClientRole.RATINGS_WORKER, gpu_info)
        workers = self_play_workers + ratings_workers
        if not workers:
            return
        self._issue_pause(workers)
        self._wait_for_pause_acks()

    def reload_weights(self, conns: List[ClientConnection], gen: Generation):
        if not conns:
            return

        logger.debug(f'Issuing reload weights (gen={gen})...')

        data = {
            'type': 'reload-weights',
            'generation': gen,
        }

        model_filename = self._controller.organizer.get_model_filename(gen)
        for conn in conns:
            with conn.socket.send_mutex():
                send_json(conn.socket.native_socket(), data)
                send_file(conn.socket.native_socket(), model_filename)

        logger.debug('Reload weights complete!')

    def handle_new_model(self, gen: Generation):
        """
        Reloads weights for all self-play workers. The act of reloading unpauses those workers.

        Additionally unpauses ratings workers that may have been paused.
        """
        gpu_info = self._controller.training_gpu_info
        self.reload_weights(self._controller.get_connections(ClientRole.SELF_PLAY_WORKER), gen)
        self._issue_unpause(self._controller.get_connections(ClientRole.RATINGS_WORKER, gpu_info))

    def _issue_unpause(self, conns: List[ClientConnection]):
        if not conns:
            return

        logger.debug(f'Unpausing {len(conns)} workers...')
        logger.debug(f'Workers: {list(map(str, conns))}')

        data = {
            'type': 'unpause',
        }

        for conn in conns:
            conn.socket.send_json(data)

    def _mark_as_paused(self, client_id: ClientId):
        with self._lock:
            self._pause_set.add(client_id)

    def _issue_pause(self, conns: List[ClientConnection]):
        logger.debug(f'Pausing {len(conns)} workers...')
        logger.debug(f'Workers: {list(map(str, conns))}')
        if not conns:
            return
        data = {'type': 'pause'}

        for conn in conns:
            try:
                self._mark_as_paused(conn.client_id)
                conn.socket.send_json(data)
            except SocketSendException:
                logger.warn(f'Error sending pause to {conn}, ignoring...')
                self._controller.handle_disconnect(conn)  # TODO: figure out what to do here

    def _wait_for_pause_acks(self):
        logger.debug(f'Waiting for pause acks...')
        with self._pauses_acked_cv:
            self._pauses_acked_cv.wait_for(lambda: not self._pause_set)
        logger.debug('All pause acks received!')
