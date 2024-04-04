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
        self._table: GpuContentionTableDict = defaultdict(lambda: defaultdict(GpuContentionTable))

        table = self.get_gpu_lock_table(controller.training_gpu_id)
        table.activate(Domain.TRAINING)

    def get_gpu_lock_table(self, gpu_id: GpuId) -> GpuContentionTable:
        with self._table_lock:
            return self._table[gpu_id.ip_address][gpu_id.device]

    def set_ratings_priority(self, elevate: bool):
        with self._table_lock:
            currently_elevated: List[GpuContentionTable] = []
            gpus_used_for_ratings: List[GpuId] = []
            for ip_address, subdict in self._table.items():
                for cuda_device, table in subdict.items():
                    if table.active(Domain.RATINGS):
                        gpu_id = GpuId(ip_address, cuda_device)
                        gpus_used_for_ratings.append(gpu_id)
                        if table.ratings_prioritized():
                            currently_elevated.append(table)

            if not gpus_used_for_ratings:
                return

            assert len(currently_elevated) <= 1, currently_elevated
            if not elevate:
                for table in currently_elevated:
                    table.deprioritize_ratings()
                return
            elif len(currently_elevated) == 1:
                # elevated table already exists, just keep it
                return

            training_gpu_id = self._controller.training_gpu_id
            preferred_gpu_ids = [gpu_id for gpu_id in gpus_used_for_ratings if
                                 gpu_id != training_gpu_id]
            if preferred_gpu_ids:
                gpu_id = preferred_gpu_ids[0]
            else:
                gpu_id = training_gpu_id

            table = self._table[gpu_id.ip_address][gpu_id.device]
            logger.debug(f'Prioritizing ratings for {gpu_id} [{table}]')
            table.prioritize_ratings()
