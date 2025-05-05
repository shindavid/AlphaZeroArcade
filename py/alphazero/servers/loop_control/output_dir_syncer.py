from __future__ import annotations

from alphazero.logic.custom_types import Generation

import logging
import os
import shutil
import tempfile
from threading import Condition, Thread
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


class OutputDirSyncer:
    """
    On cloud-compute environments, we typically have access to a local filesystem, which is wiped
    after each session, and a network filesystem, whose contents persist across sessions. The local
    filesystem is typically significantly faster.

    In such environments, we configure the network filesystem to be mounted at /workspace, while
    ~/scratch maps to the local filesystem. Our general workflow is to do all our work on the
    local filesystem, and syncing to the network filesystem to facilitate restarts and post-run
    analysis. On restarts, we copy the network filesystem contents back to the local filesystem.

    We want such syncing mechanics to happen automatically behind-the-scenes, transparent to the
    user. This is the purpose of the OutputDirSyncer class.

    On non-cloud-compute-environments, the OutputDirSyncer is a no-op.

    On cloud-compute-environments, the OutputDirSyncer behaves by syncing the local filesystem to
    the network filesystem periodically, in a separate thread. The syncing operation needs to be
    smarter than a blind rsync, because of the following deficiencies of rsync:

    1. It fails to take advantage of the sequential per-generation writing of output (e.g., if
       we already synced up to generation 100, then a sync at generation 101 does not need to
       re-sync the first 100 generations).

    2. It can copy sqlite3 files that are being written to, causing corruption.

    3. It is unaware of sequencing requirements, such as ensuring that self-play game files are
       synced before their corresponding database entries are synced (otherwise, a crash in between
       could lead to database self-play metadata entries pointing to non-existent game files).
    """
    def __init__(self, controller: LoopController):
        self._controller = controller

        self._cond = Condition()
        self._shutdown_in_progress = False

        self._last_copied_model_gen = None
        self._last_copied_checkpoint_gen = None
        self._last_copied_self_play_gen = None

        self._thread = None

        controller.register_shutdown_action(self.shutdown)

    def shutdown(self):
        if self._thread is None:
            return

        with self._cond:
            self._shutdown_in_progress = True
            self._cond.notify_all()

        self._thread.join()
        self._final_sync()

    def start(self):
        if not self._controller.on_ephemeral_local_disk_env:
            return

        dst = self._controller.persistent_organizer

        self._last_copied_model_gen = dst.get_latest_model_generation(default=0)
        self._last_copied_checkpoint_gen = dst.get_last_checkpointed_generation(default=0)

        self._thread = Thread(target=self._sync_loop, daemon=True, name='OutputDirSyncer')
        self._thread.start()
        pass

    def sync(self, max_gen: Optional[Generation]=None):
        with self._cond:
            self._sync_gen = max_gen
            self._cond.notify_all()

    def _sync_loop(self):
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._shutdown_in_progress, timeout=30)
                if self._shutdown_in_progress:
                    break
                logger.debug("OutputDirSyncer: beginning sync...")
                self._sync()
                logger.debug("OutputDirSyncer: sync complete!")
        except:
            logger.error("OutputDirSyncer: exception in sync loop", exc_info=True)
            self._controller.request_shutdown(1)

    def _sync(self, final=False):
        # First copy database files
        for pool in self._controller.db_conn_pools():
            src = pool.db_filename
            dst = os.path.join(self._controller.persistent_organizer.databases_dir,
                               os.path.basename(src))
            logger.debug("OutputDirSyncer: syncing database file: %s -> %s", src, dst)

            if not os.path.exists(src):
                continue

            # Create a unique temporary file in /tmp/
            fd, tmp_path = tempfile.mkstemp(suffix=".sqlite3", prefix="db_copy_", dir="/tmp")
            os.close(fd)  # Close the open file descriptor immediately

            try:
                # We do a two-step copy here to avoid doing an expensive remote copy of a database
                # while holding the database lock.
                with pool.db_lock:
                    shutil.copyfile(src, tmp_path)
                shutil.copyfile(tmp_path, dst)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        first_self_play_gen_to_cp = self._last_copied_model_gen

        # Next, copy model files
        gen = self._last_copied_model_gen + 1
        while True:
            src = self._controller.organizer.get_model_filename(gen)
            if not os.path.isfile(src):
                break

            dst = self._controller.persistent_organizer.get_model_filename(gen)
            logger.debug("OutputDirSyncer: syncing model file: %s -> %s", src, dst)
            shutil.copyfile(src, dst)
            gen += 1
        self._last_copied_model_gen = gen - 1

        # Next, copy checkpoint files
        gen = self._last_copied_checkpoint_gen + 1
        while True:
            src = self._controller.organizer.get_checkpoint_filename(gen)
            if not os.path.isfile(src):
                break

            dst = self._controller.persistent_organizer.get_checkpoint_filename(gen)
            logger.debug("OutputDirSyncer: syncing checkpoint file: %s -> %s", src, dst)
            shutil.copyfile(src, dst)
            gen += 1
        self._last_copied_checkpoint_gen = gen - 1

        # Next, copy self-play data
        start_gen = first_self_play_gen_to_cp
        end_gen = self._last_copied_model_gen
        if not final:
            end_gen -= 1  # more data might be coming, just wait for gen-completion or final-sync

        for gen in range(start_gen, end_gen + 1):
            src = self._controller.organizer.get_self_play_data_filename(gen)
            if not os.path.isfile(src):
                continue  # there can be gaps in self-play gens, so continue here, not break

            dst = self._controller.persistent_organizer.get_self_play_data_filename(gen)
            logger.debug("OutputDirSyncer: syncing self-play data: %s -> %s", src, dst)
            shutil.copyfile(src, dst)

        # Finally, copy log files
        src_dir = self._controller.organizer.logs_dir
        dst_dir = self._controller.persistent_organizer.logs_dir

        for root, _, files in os.walk(src_dir):
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.join(dst_dir, os.path.relpath(src, src_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)

    def _final_sync(self):
        logger.info("OutputDirSyncer: beginning final sync...")
        self._sync(final=True)
        logger.info("OutputDirSyncer: final sync complete!")
