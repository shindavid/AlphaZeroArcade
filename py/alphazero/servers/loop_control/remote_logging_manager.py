from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientId
from util.logging_util import get_logger
from util.socket_util import JsonDict

from collections import defaultdict
import io
import os
import threading
from typing import Dict


logger = get_logger()


File = io.TextIOWrapper


class RemoteLoggingManager:
    """
    The clients that connect to LoopController are configured to forward their logged messages to
    the loop controller. This manager handles those messages by writing them to files.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._lock = threading.Lock()
        self._log_files: Dict[ClientId, Dict[str, File]] = defaultdict(dict)

    def get_log_file(self, src: str, client_id: ClientId):
        with self._lock:
            subdict = self._log_files[client_id]
            f = subdict.get(src, None)
            if f is not None:
                return f
            log_dir = os.path.join(self._controller.organizer.logs_dir, src)
            os.makedirs(log_dir, exist_ok=True)
            filename = os.path.join(log_dir, f'{src}.{client_id}.log')
            f = open(filename, 'a')
            subdict[src] = f

        logger.debug(f'Opened log file: {filename}')
        return f

    def handle_log_msg(self, msg: JsonDict, conn: ClientConnection):
        line = msg['line']
        src = msg.get('src', conn.client_role.value)
        f = self.get_log_file(src, conn.client_id)
        f.write(line)
        f.flush()

    def handle_disconnect(self, conn: ClientConnection):
        with self._lock:
            subdict = self._log_files.pop(conn.client_id, None)

        if subdict is None:
            return
        for f in subdict.values():
            f.close()
            logger.debug(f'Closed log file: {f.name}')

    def close_log_file(self, msg: JsonDict, client_id: ClientId):
        close_log = msg['close_log']
        if not close_log:
            return

        src = msg['src']
        with self._lock:
            subdict = self._log_files.get(client_id, None)
            if subdict is None:
                return
            f = subdict.pop(src, None)
            if f is None:
                return
            if not subdict:
                self._log_files.pop(client_id)

        f.close()
        logger.debug(f'Closed log file: {f.name}')
