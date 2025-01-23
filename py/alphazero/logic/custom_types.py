from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List

from alphazero.logic import constants
from util.socket_util import JsonDict, Socket


Generation = int
ClientId = int
ThreadId = int
RatingTag = str


class ClientRole(Enum):
    SELF_PLAY_SERVER = 'self-play-server'
    SELF_PLAY_WORKER = 'self-play-worker'
    RATINGS_SERVER = 'ratings-server'
    RATINGS_WORKER = 'ratings-worker'
    BENCHMARKING_SERVER = 'benchmarking-server'
    BENCHMARKING_WORKER = 'benchmarking-worker'


class Domain(Enum):
    TRAINING = 'training'
    SELF_PLAY = 'self-play'
    RATINGS = 'ratings'
    SLEEPING = 'sleeping'

    @staticmethod
    def from_role(role: ClientRole):
        # TODO: add BENCHMARKING_*
        if role in (ClientRole.SELF_PLAY_SERVER, ClientRole.SELF_PLAY_WORKER):
            return Domain.SELF_PLAY
        elif role in (ClientRole.RATINGS_SERVER, ClientRole.RATINGS_WORKER,\
            ClientRole.BENCHMARKING_SERVER, ClientRole.BENCHMARKING_WORKER):
            return Domain.RATINGS
        else:
            raise ValueError(f'Unexpected role: {role}')

    @staticmethod
    def others(d: 'Domain') -> List['Domain']:
        return [d2 for d2 in Domain if d2 != d]


class ServerStatus(Enum):
    DISCONNECTED = 'disconnected'
    BLOCKED = 'blocked'
    READY = 'ready'


@dataclass(frozen=True)
class GpuId:
    ip_address: str
    device: str

    def __str__(self):
        return f'Gpu({self.device}@{self.ip_address})'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class ClientConnection:
    client_domain: Domain
    client_role: ClientRole
    client_id: ClientId
    socket: Socket
    start_timestamp: int
    client_gpu_id: GpuId
    active: bool = True
    aux: dict = field(default_factory=dict)  # maintain arbitrary state

    @property
    def ip_address(self):
        return self.client_gpu_id.ip_address

    def is_on_localhost(self):
        return self.ip_address == constants.LOCALHOST_IP

    def __str__(self):
        return f'Conn({self.client_id}, {self.client_role.value}, {self.client_gpu_id})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self._tuple() == other._tuple()

    def __hash__(self):
        return hash(self._tuple())

    def _tuple(self):
        return (self.client_domain, self.client_role, self.client_id, self.client_gpu_id)


ShutdownAction = Callable[[], None]
MsgHandler = Callable[[ClientConnection, JsonDict], bool]  # return True for loop-break
DisconnectHandler = Callable[[ClientConnection], None]
