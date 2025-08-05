from alphazero.logic import constants
from util.py_util import sha256sum
from util.socket_util import JsonDict, Socket

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Callable, List, Optional, Literal


Generation = int
ClientId = int
ThreadId = int
RatingTag = str
EvalTag = str


class ClientRole(Enum):
    SELF_PLAY_SERVER = 'self-play-server'
    SELF_PLAY_WORKER = 'self-play-worker'
    RATINGS_SERVER = 'ratings-server'
    RATINGS_WORKER = 'ratings-worker'
    EVAL_SERVER = 'eval-server'
    EVAL_WORKER = 'eval-worker'
    BENCHMARK_SERVER = 'benchmark-server'
    BENCHMARK_WORKER = 'benchmark-worker'

    @staticmethod
    def worker_roles():
        return (ClientRole.SELF_PLAY_WORKER, ClientRole.RATINGS_WORKER)

    @staticmethod
    def connection_log_level(role: 'ClientRole') -> int:
        """
        Returns the logging level for connect/disconnect info for a given client role.
        """
        if role in (ClientRole.RATINGS_WORKER, ClientRole.BENCHMARK_WORKER, ClientRole.EVAL_WORKER):
            return logging.DEBUG
        else:
            return logging.INFO


class ServerStatus(Enum):
    DISCONNECTED = 'disconnected'
    BLOCKED = 'blocked'
    READY = 'ready'


class Domain(Enum):
    TRAINING = 'training'
    SELF_PLAY = 'self-play'
    RATINGS = 'ratings'
    SELF_EVAL = 'self-eval'
    EVAL = 'eval'
    SLEEPING = 'sleeping'

    @staticmethod
    def from_role(role: ClientRole):
        if role in (ClientRole.SELF_PLAY_SERVER, ClientRole.SELF_PLAY_WORKER):
            return Domain.SELF_PLAY
        elif role in (ClientRole.RATINGS_SERVER, ClientRole.RATINGS_WORKER):
            return Domain.RATINGS
        elif role in (ClientRole.BENCHMARK_SERVER, ClientRole.BENCHMARK_WORKER):
            return Domain.BENCHMARK
        elif role in (ClientRole.EVAL_SERVER, ClientRole.EVAL_WORKER):
            return Domain.EVAL
        else:
            raise ValueError(f'Unexpected role: {role}')

    @staticmethod
    def others(d: 'Domain') -> List['Domain']:
        return [d2 for d2 in Domain if d2 != d]


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
    rating_tag: str
    active: bool = True
    aux: Any = None  # for arbitrary data

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


@dataclass
class FileToTransfer:
    """
    Contains information about a file to be transferred, including:
    - source_path: the location of the file to be transferred. It is an absolute path.
    - asset_path: the destination where the file is stored. It is a relative path from the asset
        directory specified in the session data.
    - scratch_path: the location that the file will be used or referenced. It is a relative path
        from the run directory.
    - SHA256 hash of the file.
    """
    source_path: str
    scratch_path: str
    asset_path: Optional[str] = None
    sha256_hash: Optional[str] = None

    @classmethod
    def from_src_scratch_path(cls, source_path: str, scratch_path: str, asset_path_mode: Literal['hash', 'scratch']):
        obj = cls(
            source_path=source_path,
            scratch_path=scratch_path
        )
        obj.sha256_hash = sha256sum(obj.source_path)

        if asset_path_mode == 'hash':
            obj.asset_path = obj.sha256_hash
        elif asset_path_mode == 'scratch':
            obj.asset_path = scratch_path
        else:
            raise ValueError(f"Invalid asset_path_mode: {asset_path_mode}. Must be 'hash' or 'scratch'.")

        return obj

    def to_dict(self) -> JsonDict:
        return {
            'source_path': self.source_path,
            'asset_path': self.asset_path,
            'scratch_path': self.scratch_path,
            'sha256_hash': self.sha256_hash
        }
