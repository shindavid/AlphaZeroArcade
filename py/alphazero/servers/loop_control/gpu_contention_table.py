from alphazero.logic.custom_types import Domain, GpuId

from dataclasses import dataclass
from enum import Enum
import logging
import threading
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class LockStatus(Enum):
    RELEASED = 'released'
    ACQUIRING = 'acquiring'
    ACQUIRED = 'acquired'


class ManagementStatus(Enum):
    INACTIVE = 'inactive'
    ACTIVE = 'active'
    DEACTIVATING = 'deactivating'


@dataclass
class DomainState:
    priority: int
    lock_status: LockStatus = LockStatus.RELEASED
    management_status: ManagementStatus = ManagementStatus.INACTIVE

    def clone(self) -> 'DomainState':
        return DomainState(self.priority, self.lock_status, self.management_status)

    def __str__(self):
        return f'{self.lock_status.value}/{self.management_status.value}@{self.priority}'

    def __repr__(self) -> str:
        return str(self)


PRIORITIZED_RATINGS_PRIORITY = 6
DEFAULT_TRAINING_PRIORITY = 5
DEFAULT_SELF_PLAY_PRIORITY = 4
DEFAULT_RATINGS_PRIORITY = 3
DEFAULT_SLEEPING_PRIORITY = 2
HIJACKED_SELF_PLAY_PRIORITY = 1


class GpuContentionTable:
    """
    Represents the lock status of a GPU.

    At any given time, only one domain (training, self-play, ratings) can hold the lock on a GPU.

    If multiple domains want to acquire the lock, it is awarded to the domain that has the current
    highest priority.
    """
    def __init__(self, gpu_id: GpuId):
        self._gpu_id = gpu_id
        self._states: Dict[Domain, DomainState] = {
            Domain.TRAINING: DomainState(DEFAULT_TRAINING_PRIORITY),
            Domain.SELF_PLAY: DomainState(DEFAULT_SELF_PLAY_PRIORITY),
            Domain.RATINGS: DomainState(DEFAULT_RATINGS_PRIORITY),
            Domain.SLEEPING: DomainState(DEFAULT_SLEEPING_PRIORITY),
            }

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self.activate(Domain.SLEEPING)

        threading.Thread(target=self._sleep, daemon=True, name=f'sleep-thread').start()

    @property
    def gpu_id(self) -> GpuId:
        return self._gpu_id

    def has_highest_priority(self, domain: Domain) -> bool:
        with self._lock:
            p = self._states[domain].priority
            for d in Domain.others(domain):
                if self._states[d].priority > p:
                    return False
            return True

    def active(self, domain: Domain) -> bool:
        """
        Returns True iff the management status of the domain is ACTIVE.

        Note that this is NOT the same as (not INACTIVE).
        """
        with self._lock:
            return self._active(domain)

    def inactive(self, domain: Domain) -> bool:
        """
        Returns True iff the management status of the domain is INACTIVE.

        Note that this is NOT the same as (not ACTIVE).
        """
        with self._lock:
            return self._inactive(domain)

    def activate(self, domain: Domain):
        with self._lock:
            self._states[domain].management_status = ManagementStatus.ACTIVE
            self._cond.notify_all()

    def mark_as_deactivating(self, domain: Domain):
        with self._lock:
            self._states[domain].management_status = ManagementStatus.DEACTIVATING
            self._cond.notify_all()

    def deactivate(self, domain: Domain):
        with self._lock:
            state = self._states[domain]
            state.management_status = ManagementStatus.INACTIVE
            state.lock_status = LockStatus.RELEASED
            self._cond.notify_all()

    def ratings_prioritized(self) -> bool:
        with self._lock:
            return self._states[Domain.RATINGS].priority == PRIORITIZED_RATINGS_PRIORITY

    def prioritize_ratings(self):
        with self._lock:
            self._set_priority(Domain.RATINGS, PRIORITIZED_RATINGS_PRIORITY)

    def deprioritize_ratings(self):
        with self._lock:
            self._set_priority(Domain.RATINGS, DEFAULT_RATINGS_PRIORITY)

    def hijack_self_play(self):
        with self._lock:
            self._set_priority(Domain.SELF_PLAY, HIJACKED_SELF_PLAY_PRIORITY)
            if self._states[Domain.SELF_PLAY].lock_status != LockStatus.ACQUIRED:
                return

            top_domain = self._get_top_priority_active_domain()
            assert top_domain not in (Domain.SELF_PLAY, None), top_domain
            self._acquire_lock(top_domain)

    def unhijack_self_play(self):
        with self._lock:
            self._set_priority(Domain.SELF_PLAY, DEFAULT_SELF_PLAY_PRIORITY)
            if self._states[Domain.SELF_PLAY].lock_status == LockStatus.ACQUIRED:
                return

            top_domain = self._get_top_priority_active_domain()
            if top_domain == Domain.SELF_PLAY:
                self._acquire_lock(top_domain)

    def pre_acquire_lock(self, domain: Domain):
        with self._lock:
            return self._pre_acquire_lock(domain)

    def acquire_lock(self, domain: Domain) -> bool:
        """
        This method performs a series of carefully orchestrated steps:

        0. If lock status is already ACQUIRED, then return immediately.
        1. Set the lock status to ACQUIRING. Notify all.
        2. Wait until the lock is available, OR if the management status is not ACTIVE.
        3. Sets the lock status to ACQUIRED/RELEASED based on whether management status is ACTIVE
           or not. Notify all.

        Returns True iff lock status is ACQUIRED. If this returns False, that means that the domain
        was deactivated externally (e.g., there was a server disconnect). The caller is responsible
        for handling this case appropriately.

        In order for this to work, other domains, upon receiving the notification in step 1, are
        responsible for releasing the lock if they no longer have priority. This can be
        accomplished by calling wait_for_lock_expiry() and then release_lock().

        The wait in step 2 will end due to one of these scenarios:

        A. The domain currently has priority, and so other domains immediately release the lock.
        B. The domain currently does not have priority, but its priority is later externally
           elevated.
        C. The domain currently does not have priority, but the current lock holder later
           voluntarily releases the lock or decreases its priority.
        """
        with self._lock:
            return self._acquire_lock(domain)

    def release_lock(self, domain: Domain):
        with self._lock:
            self._release_lock(domain)

    def wait_for_lock_expiry(self, domain: Domain) -> bool:
        """
        Assumes that domain currently holds the lock.

        Waits until a higher priority domain wants to acquire the lock, OR until the domain is
        not longer active.

        Returns True iff the domain is still active after the wait.

        This does not actually release the lock. The caller is responsible for calling
        release_lock() to actually release the lock.
        """
        logger.debug('..................Waiting for lock expiry for %s: %s', domain, self)
        with self._lock:
            return self._wait_for_lock_expiry(domain)

    def _release_lock(self, domain: Domain):
        if self._states[domain].lock_status == LockStatus.RELEASED:
            return
        self._states[domain].lock_status = LockStatus.RELEASED
        self._cond.notify_all()

    def _wait_for_lock_expiry(self, domain: Domain) -> bool:
        self._cond.wait_for(lambda: not self._active(domain) or self._lock_expired(domain))
        return self._active(domain)

    def _get_top_priority_active_domain(self) -> Optional[Domain]:
        pairs = [(state.priority, domain) for domain, state in self._states.items()
                 if state.management_status == ManagementStatus.ACTIVE and
                 state.lock_status != LockStatus.RELEASED]
        if not pairs:
            return None
        return max(pairs)[1]

    def _set_priority(self, domain: Domain, priority: int):
        prev_priority = self._states[domain].priority
        self._states[domain].priority = priority
        if prev_priority != priority:
            self._cond.notify_all()

    def _active(self, domain: Domain) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'
        return self._states[domain].management_status == ManagementStatus.ACTIVE

    def _inactive(self, domain: Domain) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'
        return self._states[domain].management_status == ManagementStatus.INACTIVE

    def _pre_acquire_lock(self, domain: Domain):
        logger.debug('Pre-acquiring lock for %s: %s', domain, self)
        assert self._lock.locked(), 'LockTable must be locked'
        if self._states[domain].lock_status == LockStatus.ACQUIRED:
            return True
        self._states[domain].lock_status = LockStatus.ACQUIRING
        self._cond.notify_all()

    def _acquire_lock(self, domain: Domain) -> bool:
        self._pre_acquire_lock(domain)
        self._cond.wait_for(lambda: self._states[domain].lock_status == LockStatus.ACQUIRED or
                            self._lock_available(domain) or not self._active(domain))
        active = self._active(domain)
        lock_status = LockStatus.ACQUIRED if active else LockStatus.RELEASED
        self._states[domain].lock_status = lock_status
        self._cond.notify_all()
        return active

    def _lock_available(self, domain: Domain) -> bool:
        """
        Returns True if no other domain is currently holding/releasing the lock, and if no other
        higher-priority domain wants to acquire the lock.
        """
        assert self._lock.locked(), 'LockTable must be locked'

        state = self._states[domain]
        for d in Domain.others(domain):
            s = self._states[d]
            if s.lock_status == LockStatus.ACQUIRED:
                return False
            if s.priority > state.priority and s.lock_status == LockStatus.ACQUIRING:
                return False

        return True

    def _lock_expired(self, domain: Domain) -> bool:
        """
        Returns True iff a higher-priority domain wants to acquire the lock.
        """
        assert self._lock.locked(), 'LockTable must be locked'

        state = self._states[domain]
        for d in Domain.others(domain):
            s = self._states[d]
            if s.priority > state.priority and s.lock_status == LockStatus.ACQUIRING:
                return True

        return False

    def _sleep(self):
        with self._lock:
            while True:
                self._acquire_lock(Domain.SLEEPING)
                if self._wait_for_lock_expiry(Domain.SLEEPING):
                    self._release_lock(Domain.SLEEPING)

    def __str__(self):
        g = self._gpu_id
        t = self._states[Domain.TRAINING]
        s = self._states[Domain.SELF_PLAY]
        r = self._states[Domain.RATINGS]
        u = self._states[Domain.SLEEPING]
        return f'LockTable(gpu_id={g}, training={t}, self-play={s}, ratings={r}, sleeping={u})'

    def __repr__(self) -> str:
        return str(self)