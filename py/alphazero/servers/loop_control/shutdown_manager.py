from alphazero.logic.custom_types import ShutdownAction
from util.logging_util import get_logger

import sys
import threading
from typing import List, Optional


logger = get_logger()


class ShutdownManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._shutdown_actions: List[ShutdownAction] = []
        self._shutdown_code: Optional[int] = None
        self._shutdown_in_progress = False

    def request_shutdown(self, return_code: int):
        with self._lock:
            if self._shutdown_code is None:
                self._shutdown_code = return_code
            else:
                self._shutdown_code = max(self._shutdown_code, return_code)
            self._cond.notify_all()

    def active(self):
        with self._lock:
            return not self._shutdown_in_progress

    def shutdown(self):
        with self._lock:
            code = self._shutdown_code
            self._shutdown_in_progress = True
            actions = list(self._shutdown_actions)

        for action in actions:
            try:
                action()
            except:
                logger.error('Error while shutting down', exc_info=True)
        sys.exit(code)

    def register(self, action: ShutdownAction):
        with self._lock:
            self._shutdown_actions.append(action)

    def wait_for_shutdown_request(self):
        with self._lock:
            self._cond.wait_for(lambda: self._shutdown_code is not None)
