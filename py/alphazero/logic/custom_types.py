import abc
from dataclasses import dataclass
from enum import Enum

from alphazero.logic import constants
from util.socket_util import Socket


Generation = int
ClientId = int
ThreadId = int


class ChildThreadError(Exception):
    pass


class ClientType(Enum):
    SELF_PLAY_MANAGER = 'self-play-manager'
    SELF_PLAY_WORKER = 'self-play-worker'
    RATINGS_MANAGER = 'ratings-manager'
    RATINGS_WORKER = 'ratings-worker'


@dataclass
class ClientData:
    client_type: ClientType
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

    def is_on_localhost(self):
        return self.ip_address == constants.LOCALHOST_IP

    def gpu_key(self):
        return (self.ip_address, self.cuda_device)

    def __str__(self):
        tokens = [str(self.client_type), str(self.client_id),
                  f'{self.ip_address}:{self.port}', self.cuda_device]
        tokens = [t for t in tokens if t]
        return f'ClientData({", ".join(tokens)})'


class NewModelSubscriber(abc.ABC):
    @abc.abstractmethod
    def handle_new_model(self, generation: Generation):
        pass
