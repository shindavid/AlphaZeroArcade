from dataclasses import dataclass
from enum import Enum

from alphazero.logic.custom_types import Domain
import threading
from typing import Dict


class LockStatus(Enum):
    RELEASED = 'released'
    ACQUIRING = 'acquiring'
    ACQUIRED = 'acquired'
    RELEASING = 'releasing'


class ManagementStatus(Enum):
    INACTIVE = 'inactive'
    ACTIVE = 'active'
    DEACTIVATING = 'deactivating'


@dataclass
class DomainState:
    priority: int
    lock_status: LockStatus = LockStatus.RELEASED
    management_status: ManagementStatus = ManagementStatus.INACTIVE

    def __str__(self):
        return f'{self.lock_status.value}/{self.management_status.value}@{self.priority}'

    def __repr__(self) -> str:
        return str(self)


class GpuContentionTable:
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
        self._states: Dict[Domain, DomainState] = {
            Domain.TRAINING: DomainState(3),
            Domain.SELF_PLAY: DomainState(2),
            Domain.RATINGS: DomainState(1),
            }

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

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

    def solely_dedicated_to(self, domain: Domain) -> bool:
        with self._lock:
            if self._states[domain].lock_status == LockStatus.UNUSED:
                return False
            for d in Domain.others(domain):
                if self._states[d].lock_status != LockStatus.UNUSED:
                    return False
            return True

    def unused(self) -> bool:
        with self._lock:
            return all(state.lock_status == LockStatus.UNUSED for state in self._states.values())

    def ratings_prioritized(self) -> bool:
        with self._lock:
            return self._states[Domain.RATINGS].priority == 4

    def prioritize_ratings(self):
        with self._lock:
            p_old = self._states[Domain.RATINGS].priority
            p_new = 4
            self._states[Domain.RATINGS].priority = p_new
            if p_old != p_new:
                self._cond.notify_all()

    def deprioritize_ratings(self):
        with self._lock:
            p_old = self._states[Domain.RATINGS].priority
            p_new = 1
            self._states[Domain.RATINGS].priority = p_new
            if p_old != p_new:
                self._cond.notify_all()

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
        responsible for releasing the lock if they are no longer have priority. This can be
        accomplished by calling wait_for_lock_expiry() and then release_lock().

        The wait in step 2 will end due to one of these scenarios:

        A. The domain currently has priority, and so other domains immediately release the lock.
        B. The domain currently does not have priority, but its priority is later externally
           elevated.
        C. The domain currently does not have priority, but the current lock holder later
           voluntarily releases the lock or decreases its priority.
        """
        with self._lock:
            if self._states[domain].lock_status == LockStatus.ACQUIRED:
                return True
            self._states[domain].lock_status = LockStatus.ACQUIRING
            self._cond.notify_all()
            self._cond.wait_for(
                lambda: self._lock_available(domain) or not self._active(domain))
            active = self._active(domain)
            lock_status = LockStatus.ACQUIRED if active else LockStatus.RELEASED
            self._states[domain].lock_status = lock_status
            self._cond.notify_all()
            return active

    def release_lock(self, domain: Domain):
        with self._lock:
            if self._states[domain].lock_status == LockStatus.RELEASED:
                return
            self._states[domain].lock_status = LockStatus.RELEASED
            self._cond.notify_all()

    def wait_for_lock_expiry(self, domain: Domain) -> bool:
        """
        Assumes that domain currently holds the lock.

        Waits until a higher priority domain wants to acquire the lock, OR until the domain is
        not longer active.

        Returns True iff the domain is still active after the wait.

        This does not actually release the lock. The caller is responsible for calling
        release_lock() to actually release the lock.
        """
        with self._lock:
            self._cond.wait_for(lambda: not self._active(domain) or self._lock_expired(domain))
            return self._active(domain)

    def _active(self, domain: Domain) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'
        return self._states[domain].management_status == ManagementStatus.ACTIVE

    def _inactive(self, domain: Domain) -> bool:
        assert self._lock.locked(), 'LockTable must be locked'
        return self._states[domain].management_status == ManagementStatus.INACTIVE

    def _lock_available(self, domain: Domain) -> bool:
        """
        Returns True if no other domain is currently holding/releasing the lock, and if no other
        higher-priority domain wants to acquire the lock.
        """
        assert self._lock.locked(), 'LockTable must be locked'

        state = self._states[domain]
        for d in Domain.others(domain):
            s = self._states[d]
            if s.lock_status in (LockStatus.ACQUIRED, LockStatus.RELEASING):
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

    def __str__(self):
        t = self._states[Domain.TRAINING]
        s = self._states[Domain.SELF_PLAY]
        r = self._states[Domain.RATINGS]
        return f'LockTable(training={t}, self-play={s}, ratings={r})'

    def __repr__(self) -> str:
        return str(self)