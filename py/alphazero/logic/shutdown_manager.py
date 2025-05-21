from alphazero.logic.custom_types import ShutdownAction

import logging
import sys
import threading
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)


class ShutdownManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._shutdown_actions: List[Tuple[ShutdownAction, str]] = []
        self._shutdown_code: Optional[int] = None
        self._shutdown_in_progress = False

    def request_shutdown(self, return_code: int):
        with self._lock:
            if self._shutdown_code is None:
                self._shutdown_code = return_code
            else:
                self._shutdown_code = max(self._shutdown_code, return_code)
            self._cond.notify_all()

    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_code is not None or self._shutdown_in_progress

    def active(self):
        with self._lock:
            return not self._shutdown_in_progress

    def shutdown(self):
        logger.info(f'shutdown manager shutting down with code {self._shutdown_code}')
        with self._lock:
            code = self._shutdown_code
            self._shutdown_in_progress = True
            actions = list(self._shutdown_actions)

        for action, descr in actions:
            try:
                logger.info(f'Running shutdown action: {descr}')
                action()
                logger.info(f'Shutdown action {descr} completed successfully')
            except:
                logger.error('Error while shutting down', exc_info=True)

        sys.exit(code)

    def register(self, action: ShutdownAction, descr: str):
        with self._lock:
            self._shutdown_actions.append((action, descr))

    def wait_for_shutdown_request(self):
        with self._lock:
            self._cond.wait_for(lambda: self._shutdown_code is not None)
