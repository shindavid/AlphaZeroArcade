from .loop_controller_interface import LoopControllerInterface
from alphazero.logic.custom_types import Generation, ClientConnection, ClientRole
from util.logging_util import get_logger
from util.socket_util import send_json, send_file

from typing import List


logger = get_logger()


class NetworkWeightsBroadcaster:
    """
    This manager sends network weights to worker processes.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller

    def broadcast_to_self_play_workers(self, gen: Generation):
        workers = self._controller.get_connections(role=ClientRole.SELF_PLAY_WORKER)
        self.broadcast(workers, gen)

    def broadcast(self, conns: List[ClientConnection], gen: Generation):
        if not conns:
            return

        logger.debug(f'Broadcasting weights (gen={gen}) to {conns}')

        data = {
            'type': 'reload-weights',
            'generation': gen,
        }

        model_filename = self._controller.organizer.get_model_filename(gen)
        for conn in conns:
            with conn.socket.send_mutex():
                send_json(conn.socket.native_socket(), data)
                send_file(conn.socket.native_socket(), model_filename)

        logger.debug('Weights broadcast complete!')
