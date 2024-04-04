from .gpu_contention_table import GpuContentionTable
from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import Domain, GpuId
from util.logging_util import get_logger

from collections import defaultdict
import threading
from typing import Dict, List


logger = get_logger()


IpAddress = str
CudaDevice = str
GpuContentionTableDict = Dict[IpAddress, Dict[CudaDevice, GpuContentionTable]]


class GpuContentionManager:
    """
    Manages contention for GPUs.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._table_lock = threading.Lock()
        self._table: GpuContentionTableDict = defaultdict(dict)

        table = self.get_gpu_lock_table(controller.training_gpu_id)
        table.activate(Domain.TRAINING)

    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        with self._table_lock:
            subtable = self._table[gpu_id.ip_address]
            table = subtable.get(gpu_id.device, None)
            if table is None:
                table = GpuContentionTable(gpu_id)
                subtable[gpu_id.device] = table
            return table

    def set_ratings_priority(self, elevate: bool):
        with self._table_lock:
            all_tables: List[GpuContentionTable] = []
            for subdict in self._table.values():
                for table in subdict.values():
                    all_tables.append(table)

        ratings_tables = [table for table in all_tables if table.active(Domain.RATINGS)]
        if not ratings_tables:
            return

        currently_elevated = [table for table in ratings_tables if table.ratings_prioritized()]
        assert len(currently_elevated) <= 1, currently_elevated
        if not elevate:
            for table in currently_elevated:
                table.deprioritize_ratings()
            return
        elif len(currently_elevated) == 1:
            # elevated table already exists, just keep it
            return

        ratings_tables.sort(key=lambda table:
                            (table.active(Domain.TRAINING), table.active(Domain.SELF_PLAY)))

        table = ratings_tables[0]
        logger.debug(f'Prioritizing ratings for {table}')
        table.prioritize_ratings()
