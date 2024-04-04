from .gpu_contention_table import GpuContentionTable, LockStatus
from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientId, ClientRole, \
    ClientRoleOrRoles, Domain, Generation, GpuId
from util.logging_util import get_logger
from util.socket_util import send_json, send_file, SocketSendException

from collections import defaultdict
import threading
from typing import Dict, List, Optional, Set


logger = get_logger()


# class UsageState(Enum):
#     UNUSED = 'unused'
#     IDLE = 'idle'  # only used for TRAINING domain
#     STARTING = 'starting'  # only used for SELF_PLAY/RATINGS domains
#     WAITING_LOW_PRIORITY = 'waiting-low-priority'
#     WAITING_HIGH_PRIORITY = 'waiting-high-priority'
#     RUNNING_LOW_PRIORITY = 'running-low-priority'
#     RUNNING_HIGH_PRIORITY = 'running-high-priority'

#     def is_waiting(self) -> bool:
#         return self in (UsageState.WAITING_LOW_PRIORITY, UsageState.WAITING_HIGH_PRIORITY)

#     def is_running(self) -> bool:
#         return self in (UsageState.RUNNING_LOW_PRIORITY, UsageState.RUNNING_HIGH_PRIORITY)

#     def is_high_priority(self) -> bool:
#         return self in (UsageState.WAITING_HIGH_PRIORITY, UsageState.RUNNING_HIGH_PRIORITY)


# class UsageTable:
#     def __init__(self):
#         self._states = {
#             Domain.RATINGS: UsageState.UNUSED,
#             Domain.SELF_PLAY: UsageState.UNUSED,
#             Domain.TRAINING: UsageState.UNUSED,
#             }

#     def __str__(self):
#         r = self._states[Domain.RATINGS].value
#         s = self._states[Domain.SELF_PLAY].value
#         t = self._states[Domain.TRAINING].value

#         tokens = []
#         if r != UsageState.UNUSED:
#             tokens.append(f'ratings={r}')
#         if s != UsageState.UNUSED:
#             tokens.append(f'self-play={s}')
#         if t != UsageState.UNUSED:
#             tokens.append(f'training={t}')
#         return f'UsageTable({", ".join(tokens)})'

#     def __repr__(self):
#         return str(self)

#     def __getitem__(self, domain: Domain) -> UsageState:
#         return self._states[domain]

#     def __setitem__(self, domain: Domain, status: UsageState):
#         self._states[domain] = status

#     def unused(self) -> bool:
#         return all(state == UsageState.UNUSED for state in self._states.values())

#     def waiting_count(self) -> int:
#         return sum(state.is_waiting() for state in self._states.values())

#     def running_count(self) -> int:
#         return sum(state.is_running() for state in self._states.values())

#     def high_priority_count(self) -> int:
#         return sum(state.is_high_priority() for state in self._states.values())

#     def update_from_waiting_to_running(self, conn: ClientConnection):
#         if self[conn.client_domain] == UsageState.WAITING_LOW_PRIORITY:
#             self[conn.client_domain] = UsageState.RUNNING_LOW_PRIORITY
#         elif self[conn.client_domain] == UsageState.WAITING_HIGH_PRIORITY:
#             self[conn.client_domain] = UsageState.RUNNING_HIGH_PRIORITY
#         else:
#             raise ValueError(f'Unexpected state: table={self}, conn={conn}')

#     def downgrade_priority(self, domain: Domain):
#         if self[domain] == UsageState.RUNNING_HIGH_PRIORITY:
#             self[domain] = UsageState.RUNNING_LOW_PRIORITY
#         elif self[domain] == UsageState.WAITING_HIGH_PRIORITY:
#             self[domain] = UsageState.WAITING_LOW_PRIORITY


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

    # def acquire_gpu_lock(self, domain: Domain, gpu_id: GpuId):
    #     table = self._get_table(gpu_id)
    #     logger.debug(f'Acquiring GPU lock ({domain}, {gpu_id})...: {table}')
    #     table.acquire_lock(domain)
    #     logger.debug(f'Acquired GPU lock ({domain}, {gpu_id})!: {table}')

    # def release_gpu_lock(self, domain: Domain, gpu_id: GpuId):
    #     table = self._get_table(gpu_id)
    #     logger.debug(f'Releasing GPU lock ({domain}, {gpu_id})...: {table}')
    #     table.release_lock(domain)
    #     logger.debug(f'Released GPU lock ({domain}, {gpu_id})!: {table}')

    # def mark_as_idle(self, domain: Domain, gpu_id: GpuId):
    #     table = self._get_table(gpu_id)
    #     table.set_status(domain, LockStatus.RELEASED)

    # def wait_until_gpu_priority_lost(self, domain: Domain, gpu_id: GpuId):
    #     table = self._get_table(gpu_id)
    #     table.wait_until_priority_lost(domain)

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

    # def update_latest_gen(self, gen: Generation):
    #     """
    #     Updates state to reflect completion of a train loop.
    #     """
    #     with self._table_lock:
    #         self._latest_gen = gen
    #         self._elevate_ratings_priority_if_necessary()

    # def notify_of_new_rating(self):
    #     # downgrade HIGH_PRIORITY to LOW_PRIORITY for all ratings workers
    #     with self._table_lock:
    #         for subdict in self._gpu_usage_dict.values():
    #             for table in subdict.values():
    #                 table.downgrade_priority(Domain.RATINGS)

    # def _calc_ratings_high_priority_deadline(self):
    #     deadline = self._latest_gen + self._controller.params.rating_block_rate
    #     self._ratings_high_priority_deadline = deadline

    # def _elevate_ratings_priority_if_necessary(self):
    #     """
    #     Assumes self._lock is held.
    #     """
    #     assert self._table_lock.locked(), 'Lock must be held'

    #     logger.debug('Evaluating whether to elevate ratings priority...')
    #     if self._latest_gen < self._ratings_high_priority_deadline:
    #         logger.debug(f'No elevation (latest={self._latest_gen}, '
    #                      f'deadline={self._ratings_high_priority_deadline})...')
    #         return

    #     ratings_servers = self._controller.get_connections(role=ClientRole.RATINGS_SERVER)
    #     if not ratings_servers:
    #         logger.debug(f'No elevation - no ratings servers')
    #         return

    #     candidate_gpu_ids = []
    #     for conn in ratings_servers:
    #         gpu_id = conn.client_gpu_id
    #         table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
    #         if table[Domain.RATINGS] == UsageState.UNUSED:
    #             # This means the conn just came online. Elevating here complicates logic elsewhere,
    #             # so don't elevate now. It will get elevated later.
    #             continue
    #         if table[Domain.RATINGS].is_running():
    #             logger.debug(f'No elevation - ratings in progress ({conn})')
    #             return
    #         if table[Domain.RATINGS] == UsageState.WAITING_HIGH_PRIORITY:
    #             logger.debug(f'No elevation - ratings already high priority ({conn})')
    #             return
    #         if table[Domain.TRAINING] == UsageState.UNUSED:
    #             if table[Domain.SELF_PLAY] == UsageState.UNUSED:
    #                 logger.debug(f'No elevation - unimpeded GPU ({conn})')
    #                 return
    #         candidate_gpu_ids.append(gpu_id)

    #     if not candidate_gpu_ids:
    #         logger.debug(f'No elevation - ratings servers not yet ready')
    #         return

    #     self._calc_ratings_high_priority_deadline()
    #     training_gpu_id = self._controller.training_gpu_id
    #     non_clashing_gpu_ids = [gpu_id for gpu_id in candidate_gpu_ids if gpu_id != training_gpu_id]

    #     if non_clashing_gpu_ids:
    #         # elevate one of these
    #         gpu_id: GpuId = non_clashing_gpu_ids[0]
    #         logger.debug(f'Elevating non-training GPU ({conn})')
    #         table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
    #         table[Domain.RATINGS] = UsageState.WAITING_HIGH_PRIORITY
    #         return

    #     # no non-clashing GPUs available, so elevate the training GPU
    #     assert len(candidate_gpu_ids) == 1, candidate_gpu_ids
    #     assert candidate_gpu_ids[0] == training_gpu_id, (candidate_gpu_ids, training_gpu_id)

    #     logger.debug(f'Elevating training GPU ({training_gpu_id})')
    #     table = self._gpu_usage_dict[training_gpu_id.ip_address][training_gpu_id.device]
    #     table[Domain.RATINGS] = UsageState.WAITING_HIGH_PRIORITY

    # def _get_table(self, gpu_id: GpuId) -> GpuContentionTable:
    #     with self._table_lock:
    #         return self._table[gpu_id.ip_address][gpu_id.device]

    # def _acquire_lock(self, gpu_id: GpuId, domain: Domain):
    #     assert self._lock.locked(), 'Lock must be held'

    #     self._get_table(gpu_id).set_status(domain, DomainStatus.REQUESTING_LOCK)
    #     self._cond.wait_for(lambda: self._get_table(gpu_id).get_status(domain) ==
    #                         DomainStatus.LOCK_ACQUIRED)

    # def _release_lock(self, gpu_id: GpuId, domain: Domain):
    #     assert self._lock.locked(), 'Lock must be held'

    #     self._get_table(gpu_id).set_status(domain, DomainStatus.RELEASING_LOCK)
    #     self._cond.notify_all()
    #     self._cond.wait_for(lambda: self._get_table(gpu_id).get_status(domain) == DomainStatus.IDLE)
