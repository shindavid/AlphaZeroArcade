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


class UsageState(Enum):
    UNUSED = 'unused'
    IDLE = 'idle'  # only used for TRAINING domain
    STARTING = 'starting'  # only used for SELF_PLAY/RATINGS domains
    WAITING_LOW_PRIORITY = 'waiting-low-priority'
    WAITING_HIGH_PRIORITY = 'waiting-high-priority'
    RUNNING_LOW_PRIORITY = 'running-low-priority'
    RUNNING_HIGH_PRIORITY = 'running-high-priority'

    def is_waiting(self) -> bool:
        return self in (UsageState.WAITING_LOW_PRIORITY, UsageState.WAITING_HIGH_PRIORITY)

    def is_running(self) -> bool:
        return self in (UsageState.RUNNING_LOW_PRIORITY, UsageState.RUNNING_HIGH_PRIORITY)

    def is_high_priority(self) -> bool:
        return self in (UsageState.WAITING_HIGH_PRIORITY, UsageState.RUNNING_HIGH_PRIORITY)


class UsageTable:
    def __init__(self):
        self._states = {
            Domain.RATINGS: UsageState.UNUSED,
            Domain.SELF_PLAY: UsageState.UNUSED,
            Domain.TRAINING: UsageState.UNUSED,
            }

    def __str__(self):
        r = self._states[Domain.RATINGS].value
        s = self._states[Domain.SELF_PLAY].value
        t = self._states[Domain.TRAINING].value

        tokens = []
        if r != UsageState.UNUSED:
            tokens.append(f'ratings={r}')
        if s != UsageState.UNUSED:
            tokens.append(f'self-play={s}')
        if t != UsageState.UNUSED:
            tokens.append(f'training={t}')
        return f'UsageTable({", ".join(tokens)})'

    def __repr__(self):
        return str(self)

    def __getitem__(self, domain: Domain) -> UsageState:
        return self._states[domain]

    def __setitem__(self, domain: Domain, status: UsageState):
        self._states[domain] = status

    def unused(self) -> bool:
        return all(state == UsageState.UNUSED for state in self._states.values())

    def waiting_count(self) -> int:
        return sum(state.is_waiting() for state in self._states.values())

    def running_count(self) -> int:
        return sum(state.is_running() for state in self._states.values())

    def high_priority_count(self) -> int:
        return sum(state.is_high_priority() for state in self._states.values())

    def update_from_waiting_to_running(self, conn: ClientConnection):
        if self[conn.client_domain] == UsageState.WAITING_LOW_PRIORITY:
            self[conn.client_domain] = UsageState.RUNNING_LOW_PRIORITY
        elif self[conn.client_domain] == UsageState.WAITING_HIGH_PRIORITY:
            self[conn.client_domain] = UsageState.RUNNING_HIGH_PRIORITY
        else:
            raise ValueError(f'Unexpected state: table={self}, conn={conn}')

    def downgrade_priority(self, domain: Domain):
        if self[domain] == UsageState.RUNNING_HIGH_PRIORITY:
            self[domain] = UsageState.RUNNING_LOW_PRIORITY
        elif self[domain] == UsageState.WAITING_HIGH_PRIORITY:
            self[domain] = UsageState.WAITING_LOW_PRIORITY


IpAddress = str
CudaDevice = str
GpuUsageDict = Dict[IpAddress, Dict[CudaDevice, UsageTable]]


class GpuContentionManager:
    """
    Manages contention for GPUs. Generally, the priority goes:

    training > self-play > ratings

    This manager upholds this priority through a combination of pause commands and locking
    mechanisms.

    If this priority scheme starves all ratings servers for too long, then one ratings server's
    priority is temporarily elevated. As an optimization, if this would leave a GPU temporarily
    idle, then other servers are shifted around to fill the gap.
    """
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_gen: Optional[Generation] = None
        self._pending_pause_ack_set: Set[ClientId] = set()
        self._pending_unpause_ack_set: Set[ClientId] = set()
        self._ratings_high_priority_deadline: Optional[Generation] = None
        self._gpu_usage_dict: GpuUsageDict = defaultdict(lambda: defaultdict(UsageTable))

    def setup(self):
        self._latest_gen = self._controller.organizer.get_latest_model_generation()
        self._calc_ratings_high_priority_deadline()

        gpu_id = self._controller.training_gpu_id
        self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device][Domain.TRAINING] = UsageState.IDLE

    def acquire_training_gpu_lock(self):
        with self._lock:
            gpu_id = self._controller.training_gpu_id
            logger.debug(f'Acquiring training GPU lock ({gpu_id})...')

            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            assert table[Domain.TRAINING] == UsageState.IDLE, table

            logger.debug(f'Acquiring training GPU lock ({gpu_id}) - waiting for high-priority')
            table[Domain.TRAINING] = UsageState.WAITING_HIGH_PRIORITY
            self._cond.wait_for(lambda: table.high_priority_count() == 1)

            self.pause_workers(gpu_id, lock=False)
            table[Domain.TRAINING] = UsageState.RUNNING_HIGH_PRIORITY
            assert table.running_count() == 1, (gpu_id, table)
            self._cond.notify_all()

        logger.debug(f'Acquired training GPU lock ({gpu_id})!')

    def release_training_gpu_lock(self):
        with self._lock:
            gpu_id = self._controller.training_gpu_id
            logger.debug(f'Releasing training GPU lock ({gpu_id})...')

            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            assert table[Domain.TRAINING] == UsageState.RUNNING_HIGH_PRIORITY, (gpu_id, table)
            table[Domain.TRAINING] = UsageState.IDLE
            assert table.running_count() == 0, (gpu_id, table)
            self._cond.notify_all()

        logger.debug(f'Released training GPU lock ({gpu_id})!')

    def handle_disconnect(self, conn: ClientConnection):
        gpu_id = conn.client_gpu_id
        conns = self._controller.get_connections(gpu_id)
        with self._lock:
            self._pending_pause_ack_set.discard(conn.client_id)
            self._pending_unpause_ack_set.discard(conn.client_id)
            subdict = self._gpu_usage_dict[gpu_id.ip_address]
            table = subdict[gpu_id.device]
            for domain in (Domain.SELF_PLAY, Domain.RATINGS):
                if not any(c.client_domain == domain for c in conns):
                    table[domain] = UsageState.UNUSED
                elif table[domain] == UsageState.RUNNING_HIGH_PRIORITY:
                    table[domain] = UsageState.WAITING_HIGH_PRIORITY
                elif table[domain] == UsageState.RUNNING_LOW_PRIORITY:
                    table[domain] = UsageState.WAITING_LOW_PRIORITY

            if table.unused():
                subdict.pop(gpu_id.device)
                if not subdict:
                    self._gpu_usage_dict.pop(gpu_id.ip_address)

            self._cond.notify_all()

    def pause_workers(self, gpu_id: Optional[GpuId]=None, role: Optional[ClientRoleOrRoles]=None,
                      lock=True):
        """
        Pauses workers matching the given GPU and role. Filters out workers that are in an
        UNUSED/STARTING state.

        If lock is True (default), then this grabs self._lock before issuing the pause. If it
        is False, then it assumes that self._lock is already held.
        """
        if role is None:
            role = ClientRole.worker_roles()
        workers = self._controller.get_connections(gpu_id, role)

        def pause_and_wait():
            self._issue_pause(workers)
            self._cond.wait_for(lambda: len(self._pending_pause_ack_set) == 0)

        if lock:
            with self._lock:
                pause_and_wait()
        else:
            assert self._lock.locked(), 'Lock must be held'
            pause_and_wait()

    def unpause_waiting_workers(self):
        """
        Does not naively unpause all workers at once, since that could result in contention.

        Instead, for each GPU, carefully unpauses at most one worker.

        Note that this does not necessarily unpause all the workers that were paused by the prior
        pause_workers() call.
        """
        workers = self._controller.get_connections(role=ClientRole.worker_roles())
        workers_by_gpu_id = defaultdict(list)
        for conn in workers:
            workers_by_gpu_id[conn.client_gpu_id].append(conn)

        with self._lock:
            for gpu_id, workers in workers_by_gpu_id.items():
                table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
                worker = self._get_worker_to_unpause(table, workers)
                if worker is not None:
                    self._issue_unpause([worker])
                    self._cond.wait_for(lambda: len(self._pending_unpause_ack_set) == 0)

    def start_worker(self, conn: ClientConnection, gen: Generation):
        """
        Does the work in a new thread, allowing the caller to return immediately. This is necessary
        for the parent recv-loop to receive the pause/unpause-ack's and forward them to the
        GpuContentionManager, without which this start_worker() call would deadlock.
        """
        thread = threading.Thread(target=self._start_worker_helper, name='start-worker',
                                  daemon=True, args=(conn, gen))
        thread.start()

    def handle_pause_ack(self, conn: ClientConnection):
        gpu_id = conn.client_gpu_id
        with self._lock:
            self._pending_pause_ack_set.discard(conn.client_id)
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]

            assert not table[conn.client_domain].is_high_priority(), (conn, table)
            table[conn.client_domain] = UsageState.WAITING_LOW_PRIORITY
            self._cond.notify_all()

    def handle_unpause_ack(self, conn: ClientConnection):
        gpu_id = conn.client_gpu_id
        with self._lock:
            self._pending_unpause_ack_set.discard(conn.client_id)
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]

            table.update_from_waiting_to_running(conn)
            self._cond.notify_all()

    def update_latest_gen(self, gen: Generation):
        """
        Updates state to reflect completion of a train loop.
        """
        with self._lock:
            self._latest_gen = gen
            self._elevate_ratings_priority_if_necessary()

    def notify_of_new_rating(self):
        # downgrade HIGH_PRIORITY to LOW_PRIORITY for all ratings workers
        with self._lock:
            for subdict in self._gpu_usage_dict.values():
                for table in subdict.values():
                    table.downgrade_priority(Domain.RATINGS)

    def _start_worker_helper(self, conn: ClientConnection, gen: Generation):
        try:
            if conn.client_role == ClientRole.SELF_PLAY_WORKER:
                self._start_self_play_worker(conn, gen)
            elif conn.client_role == ClientRole.RATINGS_WORKER:
                self._start_ratings_worker(conn, gen)
            else:
                raise ValueError(f'Unexpected role: {conn}')
        except:
            logger.error(f'Error starting worker ({conn}, {gen}):', exc_info=True)
            self._controller.request_shutdown(1)

    def _start_self_play_worker(self, conn: ClientConnection, gen: Generation):
        logger.debug(f'Starting self-play ({conn})...')
        gpu_id = conn.client_gpu_id

        with self._lock:
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            assert table[Domain.SELF_PLAY] == UsageState.UNUSED, (conn, table)

            table[Domain.SELF_PLAY] = UsageState.WAITING_LOW_PRIORITY
            logger.debug(f'Starting self-play ({conn}) - pausing...')
            self._issue_pause([conn])
            self._cond.wait_for(lambda: len(self._pending_pause_ack_set) == 0)

            # make sure table is still there (disconnect could have happened)
            table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
            if table is None:
                return

            logger.debug(f'Starting self-play ({conn}) - reloading weights...')
            self._controller.broadcast_weights([conn], gen)

            logger.debug(f'Starting self-play ({conn}) - waiting for 0 high-priority...')
            self._cond.wait_for(lambda: table.high_priority_count() == 0)

            # make sure table is still there (disconnect could have happened)
            table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
            if table is None:
                return

            self.pause_workers(gpu_id, ClientRole.RATINGS_WORKER, lock=False)
            logger.debug(f'Starting self-play ({conn}) - waiting for 0 high-priority/running...')
            self._cond.wait_for(lambda: table.running_count() + table.high_priority_count() == 0)

            # make sure table is still there (disconnect could have happened)
            table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
            if table is None:
                return

            logger.debug(f'Starting self-play ({conn}) - unpausing...')
            self._issue_unpause([conn])
            self._cond.wait_for(lambda: len(self._pending_unpause_ack_set) == 0)
            self._cond.notify_all()

        logger.debug(f'Started self-play ({conn})!')

    def _start_ratings_worker(self, conn: ClientConnection, gen: Generation):
        logger.debug(f'Starting ratings ({conn})...')
        gpu_id = conn.client_gpu_id

        with self._lock:
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            assert not table[Domain.RATINGS].is_running(), (conn, table)

            if not table[Domain.RATINGS].is_waiting():
                table[Domain.RATINGS] = UsageState.WAITING_LOW_PRIORITY
            logger.debug(f'Starting ratings ({conn}) - pausing...')
            self._issue_pause([conn])
            self._cond.wait_for(lambda: len(self._pending_pause_ack_set) == 0)

            # make sure table is still there (disconnect could have happened)
            table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
            if table is None:
                return

            logger.debug(f'Starting ratings ({conn}) - reloading weights...')
            self._controller.broadcast_weights([conn], gen)

            logger.debug(f'Starting ratings ({conn}) - waiting for 1 running/waiting or high-priority...')
            self._cond.wait_for(lambda: table.running_count() + table.waiting_count() == 1 or
                                table[Domain.RATINGS] == UsageState.WAITING_HIGH_PRIORITY)

           # make sure table is still there (disconnect could have happened)
            table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
            if table is None:
                return

            if table[Domain.RATINGS] == UsageState.WAITING_HIGH_PRIORITY:
                logger.debug(f'Starting ratings ({conn}) - high-priority case!')
                self.pause_workers(gpu_id, ClientRole.SELF_PLAY_WORKER, lock=False)
                logger.debug(f'Starting ratings ({conn}) - waiting for 0 running...')
                self._cond.wait_for(lambda: table.running_count() == 0)

                # make sure table is still there (disconnect could have happened)
                table = self._gpu_usage_dict.get(gpu_id.ip_address, {}).get(gpu_id.device, None)
                if table is None:
                    return

            logger.debug(f'Starting ratings ({conn}) - unpausing...')
            self._issue_unpause([conn])
            self._cond.wait_for(lambda: len(self._pending_unpause_ack_set) == 0)
            self._cond.notify_all()

        logger.debug(f'Started ratings ({conn})!')

    def _calc_ratings_high_priority_deadline(self):
        deadline = self._latest_gen + self._controller.params.rating_block_rate
        self._ratings_high_priority_deadline = deadline

    def _elevate_ratings_priority_if_necessary(self):
        """
        Assumes self._lock is held.
        """
        assert self._lock.locked(), 'Lock must be held'

        logger.debug('Evaluating whether to elevate ratings priority...')
        if self._latest_gen < self._ratings_high_priority_deadline:
            logger.debug(f'No elevation (latest={self._latest_gen}, '
                         f'deadline={self._ratings_high_priority_deadline})...')
            return

        ratings_servers = self._controller.get_connections(role=ClientRole.RATINGS_SERVER)
        if not ratings_servers:
            logger.debug(f'No elevation - no ratings servers')
            return

        candidate_gpu_ids = []
        for conn in ratings_servers:
            gpu_id = conn.client_gpu_id
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            if table[Domain.RATINGS] == UsageState.UNUSED:
                # This means the conn just came online. Elevating here complicates logic elsewhere,
                # so don't elevate now. It will get elevated later.
                continue
            if table[Domain.RATINGS].is_running():
                logger.debug(f'No elevation - ratings in progress ({conn})')
                return
            if table[Domain.RATINGS] == UsageState.WAITING_HIGH_PRIORITY:
                logger.debug(f'No elevation - ratings already high priority ({conn})')
                return
            if table[Domain.TRAINING] == UsageState.UNUSED:
                if table[Domain.SELF_PLAY] == UsageState.UNUSED:
                    logger.debug(f'No elevation - unimpeded GPU ({conn})')
                    return
            candidate_gpu_ids.append(gpu_id)

        if not candidate_gpu_ids:
            logger.debug(f'No elevation - ratings servers not yet ready')
            return

        self._calc_ratings_high_priority_deadline()
        training_gpu_id = self._controller.training_gpu_id
        non_clashing_gpu_ids = [gpu_id for gpu_id in candidate_gpu_ids if gpu_id != training_gpu_id]

        if non_clashing_gpu_ids:
            # elevate one of these
            gpu_id: GpuId = non_clashing_gpu_ids[0]
            logger.debug(f'Elevating non-training GPU ({conn})')
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            table[Domain.RATINGS] = UsageState.WAITING_HIGH_PRIORITY
            return

        # no non-clashing GPUs available, so elevate the training GPU
        assert len(candidate_gpu_ids) == 1, candidate_gpu_ids
        assert candidate_gpu_ids[0] == training_gpu_id, (candidate_gpu_ids, training_gpu_id)

        logger.debug(f'Elevating training GPU ({training_gpu_id})')
        table = self._gpu_usage_dict[training_gpu_id.ip_address][training_gpu_id.device]
        table[Domain.RATINGS] = UsageState.WAITING_HIGH_PRIORITY

    def _issue_unpause(self, conns: List[ClientConnection]):
        """
        Assumes self._lock is held.

        Caller should wait for all unpause-ack's before releasing the lock.
        """
        if not conns:
            return

        logger.debug(f'Unpausing workers: {conns}')
        data = { 'type': 'unpause', }
        for conn in conns:
            try:
                self._pending_unpause_ack_set.add(conn.client_id)
                conn.socket.send_json(data)
            except SocketSendException:
                # disconnect should happen in loop_controller recv-loop
                logger.warn(f'Error sending unpause to {conn}, ignoring...')

    def _issue_pause(self, conns: List[ClientConnection]):
        """
        Assumes self._lock is held.

        Caller should wait for all pause-ack's before releasing the lock.
        """
        assert self._lock.locked(), 'Lock must be held'
        conns = self._filter_unused(conns)
        if not conns:
            return

        logger.debug(f'Pausing workers: {conns}')
        data = {'type': 'pause'}
        for conn in conns:
            try:
                self._pending_pause_ack_set.add(conn.client_id)
                conn.socket.send_json(data)
            except SocketSendException:
                # disconnect should happen in loop_controller recv-loop
                logger.warn(f'Error sending pause to {conn}, ignoring...')

    def _get_worker_to_unpause(self, table: UsageTable,
                               workers: List[ClientConnection]) -> Optional[ClientConnection]:
        """
        If there are multiple workers for a given GPU, we cannot unpause all of them at once, since
        that would result in contention.

        To decide which one to unpause, we use the following priority scheme:

        1. If there is a high-priority worker waiting, unpause that one.
        2. If there is a self-play worker waiting, unpause that one.
        3. If there is a ratings worker waiting, unpause that one.

        Assumes self._lock is held.
        """
        assert self._lock.locked(), 'Lock must be held'
        assert len(workers) in (1, 2), workers

        if table.running_count() > 0:
            return None

        for conn in workers:
            if table[conn.client_domain] == UsageState.WAITING_HIGH_PRIORITY:
                return conn

        for conn in workers:
            if conn.client_domain == Domain.SELF_PLAY:
                if table[conn.client_domain].is_waiting():
                    return conn

        for conn in workers:
            if conn.client_domain == Domain.RATINGS:
                if table[conn.client_domain].is_waiting():
                    return conn

        return None

    def _filter_unused(self, conns: List[ClientConnection]) -> List[ClientConnection]:
        """
        There can be a race condition where a connection is freshly added by the
        ClientConnectionManager, but the GpuContentionManager does not yet have a record of it.
        This function filters out such connections. Using this in certain spots simplifies reasoning
        about edge-cases.

        Assumes that self._lock is held.
        """
        assert self._lock.locked(), 'Lock must be held'

        filtered_conns = []
        for conn in conns:
            gpu_id = conn.client_gpu_id
            table = self._gpu_usage_dict[gpu_id.ip_address][gpu_id.device]
            if table[conn.client_domain] not in (UsageState.UNUSED, UsageState.STARTING):
                filtered_conns.append(conn)
        return filtered_conns
