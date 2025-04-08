from __future__ import annotations

from .gpu_contention_table import GpuContentionTable

from alphazero.logic.agent_types import MCTSAgent, AgentRole, IndexedAgent
from alphazero.logic.custom_types import ClientConnection, ClientId, EvalTag, FileToTransfer, \
    Generation, ServerStatus
from alphazero.logic.evaluator import Evaluator, EvalUtils
from alphazero.logic.match_runner import MatchType
from alphazero.logic.ratings import WinLossDrawCounts
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from alphazero.servers.loop_control.base_manager import BaseManager, ManagerConstants
from util.socket_util import JsonDict

class BenchmarkManager(BaseManager):
  pass