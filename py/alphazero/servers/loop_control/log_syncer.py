from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection
from util.logging_util import get_logger

from collections import defaultdict
import os
import subprocess
import threading
from typing import Optional


logger = get_logger()


class LogSyncer:
    """
    LogSyncer syncs remote logs from one or more hosts to the local filesystem.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._sync_map = defaultdict(dict)  # ClientConnection -> remote_filename -> local_filename
        self._lock = threading.Lock()
        self._sync_thread = None

        controller.register_shutdown_action(self.shutdown)

    def register(self, conn: ClientConnection, remote_log_filename: str):
        """
        Registers a remote log file for syncing.
        """
        tokens = remote_log_filename.split('/')

        # Remote log format: /home/devuser/logs/{game}/{tag}/{src}/{src}-{client_id}.log
        game = tokens[-4]
        tag = tokens[-3]
        src = tokens[-2]
        file_tail = tokens[-1]

        run_params = self._controller.run_params
        assert game == run_params.game, f"Unexpected remote log filename: {remote_log_filename}"
        assert tag == run_params.tag, f"Unexpected remote log filename: {remote_log_filename}"

        # Local log format: {logs_dir}/{src}/{src}-{client_id}.log
        logs_dir = self._controller.organizer.logs_dir
        local_log_filename = os.path.join(logs_dir, src, file_tail)

        ip_addr = conn.ip_address
        with self._lock:
            logger.debug("Registered log-sync: ip=%s, remote='%s' -> '%s'", ip_addr,
                         remote_log_filename, local_log_filename)
            self._sync_map[conn][remote_log_filename] = local_log_filename

    def unregister(self, conn: ClientConnection, remote_log_filename: Optional[str] = None):
        """
        Unregisters a remote log file for syncing. If remote_log_filename is not provided, then
        unregister all logs for the given conn.

        Performs a final sync for the unregistered log(s).
        """
        ip_addr = conn.ip_address
        with self._lock:
            submap = self._sync_map.get(conn, None)
            if submap is None:
                return

            if remote_log_filename is None:
                # Final sync everything for this conn
                for rfile, lfile in submap.items():
                    logger.debug("Final sync before unregister-all for ip=%s, remote='%s'", ip_addr,
                                rfile)
                    self._sync_one_locked(conn, rfile, lfile)
                del self._sync_map[conn]
            else:
                lfile = submap.get(remote_log_filename, None)
                if lfile is None:
                    return
                logger.debug("Final sync before unregister ip=%s, remote='%s'", ip_addr,
                            remote_log_filename)
                self._sync_one_locked(conn, remote_log_filename, lfile)
                del submap[remote_log_filename]
                if not submap:
                    del self._sync_map[conn]

    def spawn_sync_thread(self):
        """
        Spawns a background thread that will snapshot self._sync_map and sync all registered logs
        (at snapshot time).

        Asserts that we don't already have a running sync thread.
        """
        with self._lock:
            # Ensure no existing sync is active
            if self._sync_thread is not None and self._sync_thread.is_alive():
                raise RuntimeError("A sync thread is already running.")

            def sync_worker():
                logger.debug("Background sync thread started.")
                self._sync_all_snapshot()
                logger.debug("Background sync thread finished.")

            self._sync_thread = threading.Thread(target=sync_worker, daemon=True)
            self._sync_thread.start()

    def wait_for_sync_thread(self):
        """
        Waits until the most recently spawned sync thread completes, if any.
        """
        with self._lock:
            thread = self._sync_thread

        if thread is None:
            logger.debug("No sync thread to wait on.")
            return

        logger.debug("Waiting for sync thread to finish...")
        thread.join()

        with self._lock:
            if self._sync_thread is thread:
                self._sync_thread = None

        logger.debug("Sync thread joined.")

    def shutdown(self):
        """
        Waits for any active sync thread, then does a final sync of all currently registered logs.
        """
        logger.debug("Performing final log sync on shutdown.")

        # 1. Wait for any in-progress sync
        with self._lock:
            thread = self._sync_thread
        if thread is not None and thread.is_alive():
            logger.debug("Waiting for sync thread before shutdown final sync...")
            thread.join()
            with self._lock:
                if self._sync_thread is thread:
                    self._sync_thread = None

        # 2. One final sync for everything still registered
        self._sync_all_snapshot()
        logger.info("Synced remote logs.")

    # ----------------------------------------------------------------
    # Internal Helper Methods
    # ----------------------------------------------------------------

    def _sync_all_snapshot(self):
        """
        Snapshots self._sync_map, then rsyncs everything in that snapshot.
        This avoids blocking register/unregister calls during sync.
        """
        with self._lock:
            # Make a copy of the entire map
            sync_map_snapshot = {
                conn: dict(file_map) for conn, file_map in self._sync_map.items()
            }

        # Now do the actual sync outside the lock
        for conn, file_map in sync_map_snapshot.items():
            for remote_file, local_file in file_map.items():
                self._sync_one(conn, remote_file, local_file)

    def _sync_one(self, conn: ClientConnection, remote_file: str, local_file: str):
        """
        Rsync a single file (outside the lock). Uses append mode, no compress.
        """
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        ip_addr = conn.ip_address
        cmd = [
            "rsync", "-av", "--append", "--no-compress",
            f"devuser@{ip_addr}:{remote_file}",
            local_file
        ]
        cmd_str = " ".join(cmd)
        logger.debug(f"Syncing remote '{remote_file}' from {ip_addr} -> '{local_file}'")
        try:
            subprocess.run(cmd, check=True, capture_output=True, preexec_fn=os.setsid)
        except subprocess.CalledProcessError as e:
            logger.error("Rsync failed [%s]: %s", cmd_str, e.stderr.decode('utf-8'))

    def _sync_one_locked(self, conn: ClientConnection, remote_file: str, local_file: str):
        """
        A helper to do a one-file sync *while still holding the lock*.
        Used for final sync in unregister().
        Because it's presumably quick, we do it in the lock.
        Alternatively, you could also snapshot just that single file and do it outside the lock.
        """
        # We can do the actual rsync outside the lock if you prefer,
        # but here's the simplest approach:
        self._sync_one(conn, remote_file, local_file)
