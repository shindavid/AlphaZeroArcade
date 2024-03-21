from dataclasses import dataclass
from enum import Enum
from typing import Callable

from alphazero.logic import constants
from util.socket_util import JsonDict, Socket


Generation = int
ClientId = int
ThreadId = int


class ClientRole(Enum):
    SELF_PLAY_SERVER = 'self-play-server'
    SELF_PLAY_WORKER = 'self-play-worker'
    RATINGS_SERVER = 'ratings-server'
    RATINGS_WORKER = 'ratings-worker'


@dataclass
class GpuId:
    ip_address: str
    device: str

    def __str__(self):
        return f'Gpu({self.device}@{self.ip_address})'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class ClientConnection:
    client_role: ClientRole
    client_id: ClientId
    socket: Socket
    start_timestamp: int
    client_gpu_id: GpuId

    @property
    def ip_address(self):
        return self.client_gpu_id.ip_address

    def is_on_localhost(self):
        return self.ip_address == constants.LOCALHOST_IP

    def __str__(self):
        return f'Conn({self.client_id}, {self.client_role.value}, {self.client_gpu_id})'

    def __repr__(self):
        return str(self)


ShutdownAction = Callable[[], None]
MsgHandler = Callable[[ClientConnection, JsonDict], bool]  # return True for loop-break
DisconnectHandler = Callable[[ClientConnection], None]
