from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.str_util import make_args_str

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import os


class Agent(ABC):
    """
    Base class for agents. All agents must implment make_player_str() for generating
    command-line arguments for the player.
    """
    @abstractmethod
    def make_player_str(self, organizer: DirectoryOrganizer) -> str:
        pass


@dataclass(frozen=True)
class MCTSAgent(Agent):
    gen: int = 0
    n_iters: Optional[int] = None
    set_temp_zero: bool = None
    tag: str = None
    binary: str = None
    model: str = None

    def make_player_str(self, run_dir) -> str:
        player_args = {
            '--type': 'MCTS-C',
            '--name': f'MCTS-{self.gen}-{self.n_iters}',
            '-n': 1,
        }

        if self.n_iters is not None:
            player_args['-i'] = self.n_iters

        if self.gen == 0:
            player_args['--no-model'] = None
        else:
            player_args['-m'] = os.path.join(run_dir, self.model)

        if self.set_temp_zero:
            player_args['--starting-move-temp'] = 0
            player_args['--ending-move-temp'] = 0

        return make_args_str(player_args)


@dataclass(frozen=True)
class ReferenceAgent(Agent):
    type_str: str
    strength_param: str
    strength: int
    tag: str = None

    def make_player_str(self, organizer: DirectoryOrganizer) -> str:
        assert organizer.tag == self.tag

        player_args = {
            '--type': self.type_str,
            '--name': f'{self.type_str}-{self.strength}',
            f'{self.strength_param}': self.strength,
        }
        return make_args_str(player_args)


ArenaIndex = int  # index of an agent in an Arena
AgentDBId = int  # id in agents table of the database


class AgentRole(Enum):
    BENCHMARK = 'benchmark'
    TEST = 'test'


@dataclass
class IndexedAgent:
    """
    A dataclass for storing an agent with auxiliary info.

    - index refers to the index of the agent in the Arena's data structures.
    - db_id is the id of the agent in the database. This might be set after initial creation.
    """
    agent: Agent
    index: ArenaIndex
    role: AgentRole
    db_id: Optional[AgentDBId] = None


BenchmarkCommittee = np.ndarray # committee[k] == True iff iagent with index==k is in committee
