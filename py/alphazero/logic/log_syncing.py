from alphazero.logic.shutdown_manager import ShutdownManager
from util.logging_util import get_logger

from dataclasses import dataclass
import os
import threading


logger = get_logger()


DEFAULT_CHUNK_SIZE = 1024
DEFAULT_SYNC_PERIOD_SEC = 3


@dataclass
class SyncStatus:
    destination: str
    closed: threading.Event = threading.Event()


class LogSyncer:
    """
    A LogSyncer continuously syncs 1 or more source files to corresponding destination files.
    """
    def __init__(self, shutdown_manager: ShutdownManager, chunk_size=DEFAULT_CHUNK_SIZE,
                 sync_period_sec=DEFAULT_SYNC_PERIOD_SEC):
        shutdown_manager.register(self._shutdown)

        self._shutdown_manager = shutdown_manager
        self._chunk_size = chunk_size
        self._sync_period_sec = sync_period_sec

        self._lock = threading.Lock()
        self._shutdown_requested = False
        self._map = {}  # source -> SyncStatus

    def sync(self, source: str, destination: str):
        with self._lock:
            if self._shutdown_requested:
                return

            status = self._map.get(source, None)
            if status is not None:
                assert status.destination == destination, \
                    f"Registration clash {source} [{status.destination} != {destination}]"
                return

            logger.debug("[log_sync %s -> %s] Starting sync.", source, destination)
            status = SyncStatus(destination)
            self._map[source] = status
            threading.Thread(target=self._sync_loop, args=(status, source, destination)).start()

    def close(self, source: str):
        with self._lock:
            status = self._map.pop(source)
            logger.debug("[log_sync %s -> %s] Closing sync.", source, status.destination)
            status.closed.set()

    def _shutdown(self):
        with self._lock:
            self._shutdown_requested = True
            for status in self._map.values():
                status.closed.set()

    def _sync_loop(self, status: SyncStatus, source: str, destination: str):
        """
        A straightforward implementation would just rely on rsync. However, that would fail to fully
        exploit the fact that logs are only appended to.

        This implementation instead heuristically validates that the source and destination paths
        match up to byte D, and if so, only appends the last (S-D) bytes of the source to the
        destination, where S and D are the sizes of the source and destination files, respectively.

        Currently, this only works when we have access to both the source and destination
        filesystems. If and when we want to sync logs to a remote server, we will need to modify
        this implementation. I have not yet decided on how to set up ssh permissions, and that will
        dictate whether we need to "pull" or "push", which will affect details of the
        implementation.
        """
        try:
            logger.debug("[log_sync %s -> %s] Entered sync loop.", source, destination)

            directory = os.path.dirname(destination)
            if directory:
                os.makedirs(directory, exist_ok=True)

            while True:
                should_break = status.closed.is_set()

                src_size = LogSyncer._get_file_size(source)
                dst_size = LogSyncer._get_file_size(destination)

                logger.debug("[log_sync %s -> %s] src:%s dst:%s brk:%s", source, destination,
                            src_size, dst_size, should_break)

                if src_size == 0:
                    logger.debug("[log_sync %s -> %s] Empty/non-existent source.", source, destination)
                elif self._can_quick_sync(source, destination, src_size, dst_size):
                    self._quick_sync(source, destination, src_size, dst_size)
                else:
                    with open(source, "rb") as src, open(destination, "wb") as dst:
                        dst.write(src.read())

                if should_break:
                    break
                status.closed.wait(self._sync_period_sec)
            logger.debug("[log_sync %s -> %s] Exited sync loop.", source, destination)
        except:
            logger.error("[log_sync %s -> %s] Unexpected error in sync loop:", source, destination,
                         exc_info=True)
            self._shutdown_manager.request_shutdown(1)

    @staticmethod
    def _get_file_size(file_path):
        try:
            return os.path.getsize(file_path)
        except FileNotFoundError:
            return 0

    def _can_quick_sync(self, source: str, destination: str, src_size: int, dst_size: int):
        if dst_size < 2 * self._chunk_size or src_size < 2 * self._chunk_size:
            logger.debug("[log_sync %s -> %s] Small", source, destination)
            return False

        if dst_size > src_size:
            logger.warning("[log_sync %s -> %s] Destination is larger than source.",
                           source, destination)
            return False

        try:
            with open(source, "rb") as src, open(destination, "rb") as dst:
                src_prefix = src.read(self._chunk_size)
                dst_prefix = dst.read(self._chunk_size)
                if src_prefix != dst_prefix:
                    logger.warning("[log_sync %s -> %s] Prefix mismatch.", source, destination)
                    return False

                suffix_start = dst_size - self._chunk_size

                src.seek(suffix_start)
                src_suffix = src.read(self._chunk_size)

                dst.seek(suffix_start)
                dst_suffix = dst.read(self._chunk_size)

                if src_suffix != dst_suffix:
                    logger.warning("[log_sync %s -> %s] Suffix mismatch.", source, destination)
                    return False

        except Exception as e:
            logger.error("[log_sync %s -> %s] Error checking for quick-sync: %s",
                         source, destination, e)
            return False

        return True

    def _quick_sync(self, source: str, destination: str, src_size: int, dst_size: int):
        if src_size > dst_size:
            logger.debug("[log_sync] Syncing from offset %s to %s.", dst_size, src_size)
            with open(source, "rb") as src, open(destination, "ab") as dst:
                src.seek(dst_size)
                dst.write(src.read())
        else:
            logger.debug("[log_sync] No new data to sync.")
