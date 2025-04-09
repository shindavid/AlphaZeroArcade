from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import MCTSAgent, AgentRole, IndexedAgent
from alphazero.logic.benchmarker import Benchmarker
from alphazero.logic.custom_types import ClientConnection, ClientId, EvalTag, FileToTransfer, \
    Generation, ServerStatus

from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.base_manager import BaseManager, ManagerConstants
from util.socket_util import JsonDict

from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Optional


class MatchRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    PENDING = 'pending'


class BenchmarkRequestStatus(Enum):
    REQUESTED = 'requested'
    COMPLETE = 'complete'
    FAILED = 'failed'


@dataclass
class MatchStatus:
    opponent_gen: Generation
    n_games: int
    status: MatchRequestStatus


@dataclass
class BenchmarkStatus:
    mcts_gen: Generation
    owner: Optional[ClientId] = None
    ix_match_status: dict[int, MatchStatus] = field(default_factory=dict) # ix -> MatchStatus
    status: Optional[BenchmarkRequestStatus] = None
    committee: Optional[bool] = None


@dataclass
class ServerAux:
    """
    Auxiliary data stored per server connection.
    """
    status_cond: threading.Condition = field(default_factory=threading.Condition)
    status: ServerStatus = ServerStatus.BLOCKED
    ix: Optional[int] = None  # test agent index

class BenchmarkManager(BaseManager):
    ServerAuxClass = ServerAux

    MANAGER_CONSTANTS = ManagerConstants(
        server_name='benchmark-server',
        worker_name='benchmark-worker')

    def __init__(self, controller, tag: str):
        super().__init__(controller, tag)
        self.benchmarker = Benchmarker(self._controller.organizer)
        self._status_dict: dict[int, BenchmarkStatus] = {} # ix -> EvalStatus

    def _set_priority(self):
        latest_gen = self._controller.organizer.get_latest_model_generation()

    def _load_past_data(self):
        benchmark_rating_data = self.benchmarker.read_ratings_from_db()
