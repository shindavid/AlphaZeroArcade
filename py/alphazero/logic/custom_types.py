from dataclasses import dataclass
from enum import Enum
import socket


Generation = int
ClientId = int
ThreadId = int


class ChildThreadError(Exception):
    pass


class ClientType(Enum):
    SELF_PLAY_MANAGER = 'self-play-manager'
    SELF_PLAY_WORKER = 'self-play-worker'


@dataclass
class ClientData:
    client_type: ClientType
    client_id: ClientId
    sock: socket.socket
    start_timestamp: int
    cuda_device: str  # empty str if no cuda device

    @property
    def ip_address(self):
        return self.sock.getsockname()[0]

    @property
    def port(self):
        return self.sock.getsockname()[1]

    def is_on_localhost(self):
        return self.ip_address == '127.0.0.1'

    def __str__(self):
        tokens = [str(self.client_type), str(self.client_id),
                  f'{self.ip_address}:{self.port}', self.cuda_device]
        tokens = [t for t in tokens if t]
        return f'ClientData({", ".join(tokens)})'
