from dataclasses import dataclass
from enum import Enum
from typing import Callable

from alphazero.logic import constants
from util.socket_util import JsonDict, Socket


Generation = int
ClientId = int
ThreadId = int


class ClientRole(Enum):
    SELF_PLAY_MANAGER = 'self-play-manager'
    SELF_PLAY_WORKER = 'self-play-worker'
    RATINGS_MANAGER = 'ratings-manager'
    RATINGS_WORKER = 'ratings-worker'


@dataclass
class GpuInfo:
    ip_address: str
    device: str


@dataclass
class ClientConnection:
    client_role: ClientRole
    client_id: ClientId
    socket: Socket
    start_timestamp: int
    cuda_device: str  # empty str if no cuda device

    @property
    def ip_address(self):
        return self.socket.getsockname()[0]

    @property
    def port(self):
        return self.socket.getsockname()[1]

    @property
    def client_gpu_info(self):
        return GpuInfo(self.ip_address, self.cuda_device)

    def is_on_localhost(self):
        return self.ip_address == constants.LOCALHOST_IP

    def __str__(self):
        tokens = [str(self.client_role), str(self.client_id),
                  f'{self.ip_address}:{self.port}', self.cuda_device]
        tokens = [t for t in tokens if t]
        return f'ClientConnection({", ".join(tokens)})'


ShutdownAction = Callable[[], None]
MsgHandler = Callable[[ClientConnection, JsonDict], bool]  # return True for loop-break
DisconnectHandler = Callable[[ClientConnection], None]
