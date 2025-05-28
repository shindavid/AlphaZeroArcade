from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.socket_util import JsonDict
from util.str_util import make_args_str

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set
import os


class Agent(ABC):
    """
    Base class for agents. All agents must implment make_player_str() for generating
    command-line arguments for the player.
    """
    @abstractmethod
    def make_player_str(self, run_dir: str, args=None) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> JsonDict:
        pass

    @property
    @abstractmethod
    def level(self) -> int:
        """
        Gen for MCTS agents, strength for reference agents.
        """
        pass


@dataclass(frozen=True)
class MCTSAgent(Agent):
    gen: int = 0
    n_iters: Optional[int] = None
    set_temp_zero: bool = False
    tag: Optional[str] = None
    binary: Optional[str] = None
    model: Optional[str] = None

    def make_player_str(self, run_dir, args=None) -> str:
        name_tokens = ['MCTS', str(self.gen)]
        if self.n_iters is not None:
            name_tokens.append(str(self.n_iters))
        name = '-'.join(name_tokens)
        player_args = {
            '--type': 'MCTS-C',
            '--name': name,
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

        if args:
            player_args.update(args)

        return make_args_str(player_args)

    def to_dict(self) -> JsonDict:
        return {
            'gen': self.gen,
            'n_iters': self.n_iters,
            'set_temp_zero': self.set_temp_zero,
            'tag': self.tag,
            'binary': self.binary,
            'model': self.model
        }

    def __str__(self) -> str:
        return f'MCTSAgent-gen-{self.gen}'

    @property
    def level(self) -> int:
        return self.gen

@dataclass(frozen=True)
class ReferenceAgent(Agent):
    type_str: str
    strength_param: str
    strength: int
    tag: Optional[str] = None
    binary: Optional[str] = None

    def make_player_str(self, run_dir: str, args=None) -> str:
        player_args = {
            '--type': self.type_str,
            '--name': f'{self.type_str}-{self.strength}',
            f'{self.strength_param}': self.strength,
        }
        return make_args_str(player_args)

    def to_dict(self) -> JsonDict:
        return {
            'type_str': self.type_str,
            'strength_param': self.strength_param,
            'strength': self.strength,
            'tag': self.tag,
            'binary': self.binary
        }

    def __str__(self) -> str:
        return f'ReferenceAgent{self.strength_param}-{self.strength}'

    @property
    def level(self) -> int:
        return self.strength

ArenaIndex = int  # index of an agent in an Arena
AgentDBId = int  # id in agents table of the database


class AgentRole(Enum):
    BENCHMARK = 'benchmark'
    TEST = 'test'

    @staticmethod
    def to_str(role_set: Set['AgentRole']) -> str:
        return ','.join(x.value for x in sorted(role_set, key=lambda r: r.value))

    @staticmethod
    def from_str(str_roles: str) -> Set['AgentRole']:
        return {AgentRole(role) for role in str_roles.split(',') if role}


@dataclass
class IndexedAgent:
    """
    A dataclass for storing an agent with auxiliary info.

    - index refers to the index of the agent in the Arena's data structures.
    - roles is a set of roles that the agent can play in the arena. An agent could be both a
      benchmark and a test agent.
    - db_id is the id of the agent in the database. This might be set after initial creation.
    """
    agent: Agent
    index: ArenaIndex
    roles: Set[AgentRole]
    db_id: Optional[AgentDBId] = None

    def to_dict(self):
        agent_type = 'MCTS' if isinstance(self.agent, MCTSAgent) else 'Reference'
        return {'agent': self.agent.to_dict(),
                'index': self.index,
                'roles': self.roles.to_str(),
                'db_id': self.db_id,
                'type': agent_type}


class MatchType(Enum):
    BENCHMARK = 'benchmark'
    EVALUATE = 'evaluate'


@dataclass
class Match:
    agent1: Agent
    agent2: Agent
    n_games: int
    type: MatchType
