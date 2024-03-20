from alphazero.logic.custom_types import ClientConnection, ClientId
from .loop_controller_interface import LoopControllerInterface
from util.logging_util import get_logger
from util.socket_util import JsonDict

import os
import threading


logger = get_logger()


class RemoteLoggingManager:
    """
    The clients that connect to LoopController are configured to forward their logged messages to
    the loop controller. This manager handles those messages by writing them to files.

    TODO: change _log_files to be a dict of (parent-)client-id -> src -> file. Then, on (parent)
    disconnects, we can close all the files for that client-id. For worker-disconnects, we can have
    the parent explicitly send a message to the loop controller to close the log file. As it stands
    now, we don't have a good way to close log files when the owning process disconnects.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._lock = threading.Lock()
        self._log_files = dict()

    def get_log_file(self, src: str, client_id: ClientId):
        key = (src, client_id)
        with self._lock:
            f = self._log_files.get(key, None)
            if f is not None:
                return f
            log_dir = os.path.join(self._controller.organizer.logs_dir, src)
            os.makedirs(log_dir, exist_ok=True)
            filename = os.path.join(log_dir, f'{src}.{client_id}.log')
            f = open(filename, 'a')
            self._log_files[key] = f

        logger.info(f'Opened log file: {filename}')
        return f

    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        line = msg['line']
        src = msg.get('src', conn.client_role.value)
        f = self.get_log_file(src, conn.client_id)
        f.write(line)
        f.flush()

    def handle_disconnect(self, conn: ClientConnection):
        """
        TODO: implement me. See TODO at top of class.
        """
        pass
