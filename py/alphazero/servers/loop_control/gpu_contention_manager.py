from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.custom_types import Domain, GpuId

from collections import defaultdict
import logging
import threading
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loop_controller import LoopController


logger = logging.getLogger(__name__)


IpAddress = str
CudaDevice = str
GpuContentionTableDict = Dict[IpAddress, Dict[CudaDevice, GpuContentionTable]]


class GpuContentionManager:
    """
    Manages contention for GPUs.
    """
    def __init__(self, controller: LoopController):
        self._self_play_hijacked = True
        self._controller = controller
        self._table_lock = threading.Lock()
        self._table: GpuContentionTableDict = defaultdict(dict)

        table = self.get_gpu_lock_table(controller.default_training_gpu_id)
        table.activate(Domain.TRAINING)

    def get_gpu_lock_table_for_training(self) -> GpuContentionTable:
        """
        By default, gets the lock table for the default training GPU.

        However, if a different domain currently claims higher priority for that GPU, and if
        there is a second GPU on this host for which the TRAINING domain has higher priority, then
        that second GPU's lock table is returned instead.

        This switcheroo should only kick-in in the case where the 3 domains (TRAINING, SELF_PLAY,
        RATINGS) are competing for 2 GPUs on the same machine. Without the switcheroo, one GPU
        can inefficiently remain idle.
        """
        gpu_id = self._controller.default_training_gpu_id
        with self._table_lock:
            subtable = self._table[gpu_id.ip_address]
            table = subtable[gpu_id.device]
            if not table.has_highest_priority(Domain.TRAINING):
                for other_table in subtable.values():
                    if other_table.gpu_id != gpu_id:
                        assert other_table.has_highest_priority(Domain.TRAINING), other_table
                        logger.debug('Performing training switcheroo: %s -> %s', table, other_table)
                        return other_table
            return table

    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        with self._table_lock:
            subtable = self._table[gpu_id.ip_address]
            table = subtable.get(gpu_id.device, None)
            if table is None:
                table = GpuContentionTable(gpu_id)

                if self._self_play_hijacked:
                    table.hijack_self_play()
                subtable[gpu_id.device] = table
            return table

    def set_domain_priority(self, domain: Domain, elevate: bool):
        all_tables = self._get_all_tables()

        # When a worker is disconnected, the domain status becomes DEACTIVATING. We want to set
        # priority for tables that are not inactive.
        domain_tables = [table for table in all_tables if not table.inactive(domain)]
        if not domain_tables:
            return

        currently_elevated = [table for table in domain_tables if table.domain_prioritized(domain)]

        # TODO: this assert fails sometimes, figure out a fix
        assert len(currently_elevated) <= 1, currently_elevated
        if not elevate:
            for table in currently_elevated:
                table.deprioritize_domain(domain)
            return
        elif len(currently_elevated) == 1:
            # elevated table already exists, just keep it
            return

        domain_tables.sort(key=lambda table:
                            (table.active(Domain.TRAINING), table.active(Domain.SELF_PLAY)))

        table = domain_tables[0]
        logger.debug('Prioritizing ratings for %s', table)
        table.prioritize_domain(domain)

    def hijack_all_self_play_tables(self):
        self._self_play_hijacked = True
        all_tables = self._get_all_tables()
        for table in all_tables:
            table.hijack_self_play()

    def unhijack_all_self_play_tables(self):
        self._self_play_hijacked = False
        all_tables = self._get_all_tables()
        for table in all_tables:
            table.unhijack_self_play()

    def _get_all_tables(self) -> List[GpuContentionTable]:
        with self._table_lock:
            all_tables: List[GpuContentionTable] = []
            for subdict in self._table.values():
                for table in subdict.values():
                    all_tables.append(table)
            return all_tables