from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientId, ClientRole, \
    ClientRoleOrRoles, Domain, Generation, GpuId
from util.logging_util import get_logger
from util.socket_util import send_json, send_file, SocketSendException

from collections import defaultdict
from enum import Enum
import threading
from typing import Dict, List, Optional, Set


logger = get_logger()


class DomainStatus(Enum):
    UNUSED = 'unused'
    IDLE = 'idle'
    REQUESTING_LOCK = 'requesting-lock'
    LOCK_ACQUIRED = 'locked'
    RELEASING_LOCK = 'releasing-lock'


class LockTable:
    """
    Represents the lock status of a GPU.

    At any given time, only one domain (training, self-play, ratings) can hold the lock on a GPU.

    The lock is awarded to the domain that has the current highest priority.

    Default priority values (higher values = higher priority):

    training = 3
    self-play = 2
    ratings = 1

    If ratings have been starved for too long, then ratings is temporarily elevated to priority 4.
    """
    def __init__(self):
        self._states = {
            Domain.TRAINING: DomainStatus.UNUSED,
            Domain.SELF_PLAY: DomainStatus.UNUSED,
            Domain.RATINGS: DomainStatus.UNUSED,
            }

        self._priorities = {
            Domain.TRAINING: 3,
            Domain.SELF_PLAY: 2,
            Domain.RATINGS: 1,
            }

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def unused(self) -> bool:
        return all(state == DomainStatus.UNUSED for state in self._states.values())

    def prioritize_ratings(self):
        self._priorities[Domain.RATINGS] = 4

    def deprioritize_ratings(self):
        self._priorities[Domain.RATINGS] = 1

    def acquire_lock(self, domain: Domain):
        with self._lock:
            assert self._states[domain] != DomainStatus.LOCK_ACQUIRED, self
            self._states[domain] = DomainStatus.REQUESTING_LOCK
            self._cond.notify_all()
            self._cond.wait_for(lambda: self._has_priority(domain))

            notify = False
            for d in Domain.others(domain):
                if self._states[d] == DomainStatus.LOCK_ACQUIRED:
                    self._states[d] = DomainStatus.RELEASING_LOCK
                    notify = True
            if notify:
                self._cond.notify_all()

            self._cond.wait_for(lambda: self._ready_to_acquire_lock(domain))
            self._states[domain] = DomainStatus.LOCK_ACQUIRED
            self._cond.notify_all()

    def release_lock(self, domain: Domain):
        with self._lock:
            self._states[domain] = DomainStatus.UNUSED
            self._cond.notify_all()
            # self._cond.wait_for(lambda: not self._lock_acquisition_pending())

    def _lock_acquisition_pending(self) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'

        if DomainStatus.LOCK_ACQUIRED in self._states.values():
            return False
        if DomainStatus.REQUESTING_LOCK in self._states.values():
            return True
        return False

    def _ready_to_acquire_lock(self, domain: Domain) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'

        for d in Domain.others(domain):
            if self._states[d] in (DomainStatus.LOCK_ACQUIRED, DomainStatus.RELEASING_LOCK):
                return False
        return self._has_priority(domain)

    def _has_priority(self, domain: Domain) -> bool:
        """
        If another non-{unused/idle} domain has higher priority, then return False. Otherwise,
        returns True.
        """
        assert self._lock.locked(), 'LockTable must be locked'

        for d in Domain.others(domain):
            if self._priorities[d] > self._priorities[domain]:
                if self._states[d] not in (DomainStatus.UNUSED, DomainStatus.IDLE):
                    return False

        return True

    def wait_until_priority_lost(self, domain: Domain):
        with self._lock:
            self._cond.wait_for(lambda: not self._has_priority(domain))

    def set_status(self, domain: Domain, status: DomainStatus):
        with self._lock:
            self._states[domain] = status
            self._cond.notify_all()

    # def get_status(self, domain: Domain) -> DomainStatus:
    #     return self._states[domain]

    def __str__(self):
        ts = self._states[Domain.TRAINING].value
        ss = self._states[Domain.SELF_PLAY].value
        rs = self._states[Domain.RATINGS].value

        tp = self._priorities[Domain.TRAINING]
        sp = self._priorities[Domain.SELF_PLAY]
        rp = self._priorities[Domain.RATINGS]
        return f'LockTable(training={ts}@{tp}, self-play={ss}@{sp}, ratings={rs}@{rp})'

    def __repr__(self) -> str:
        return str(self)


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
LockTableDict = Dict[IpAddress, Dict[CudaDevice, LockTable]]


class GpuContentionManager:
    """
    Manages contention for GPUs.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._table_lock = threading.Lock()
        self._latest_gen: Optional[Generation] = None
        self._pending_pause_ack_set: Set[ClientId] = set()
        self._pending_unpause_ack_set: Set[ClientId] = set()
        self._ratings_high_priority_deadline: Optional[Generation] = None
        self._lock_table_dict: LockTableDict = defaultdict(lambda: defaultdict(LockTable))

    def setup(self):
        self._latest_gen = self._controller.organizer.get_latest_model_generation()
        self._calc_ratings_high_priority_deadline()

        table = self._get_table(self._controller.training_gpu_id)
        table.set_status(Domain.TRAINING, DomainStatus.IDLE)

    def acquire_gpu_lock(self, domain: Domain, gpu_id: GpuId):
        table = self._get_table(gpu_id)
        logger.debug(f'Acquiring GPU lock ({domain}, {gpu_id})...: {table}')
        table.acquire_lock(domain)
        logger.debug(f'Acquired GPU lock ({domain}, {gpu_id})!: {table}')

    def release_gpu_lock(self, domain: Domain, gpu_id: GpuId):
        table = self._get_table(gpu_id)
        logger.debug(f'Releasing GPU lock ({domain}, {gpu_id})...: {table}')
        table.release_lock(domain)
        logger.debug(f'Released GPU lock ({domain}, {gpu_id})!: {table}')

    def mark_as_idle(self, domain: Domain, gpu_id: GpuId):
        table = self._get_table(gpu_id)
        table.set_status(domain, DomainStatus.IDLE)

    def wait_until_gpu_priority_lost(self, domain: Domain, gpu_id: GpuId):
        table = self._get_table(gpu_id)
        table.wait_until_priority_lost(domain)

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

    def _calc_ratings_high_priority_deadline(self):
        deadline = self._latest_gen + self._controller.params.rating_block_rate
        self._ratings_high_priority_deadline = deadline

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

    def _get_table(self, gpu_id: GpuId) -> LockTable:
        with self._table_lock:
            return self._lock_table_dict[gpu_id.ip_address][gpu_id.device]

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
