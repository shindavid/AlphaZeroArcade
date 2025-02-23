from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.str_util import make_args_str

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Dict


class Agent(ABC):
    """
    Base class for agents. All agents must implment make_player_str() for generating
    command-line arguments for the player.
    """
    @abstractmethod
    def make_player_str(self, organizer: DirectoryOrganizer):
        pass

    @property
    @abstractmethod
    def version(self):
        """
        The numerical version of the agent that identifies it from its peers either created from
        the same run, e.g. a generation number or different strength value for a reference agent.
        It is used to find the left and right versions and interpolate for the estimated rating.
        """
        pass

@dataclass(frozen=True)
class MCTSAgent(Agent):
    gen: int = 0
    n_iters: int = 0
    set_temp_zero: bool = None
    binary_filename: str = None
    model_filename: str = None

    def __repr__(self):
        return f'{self.gen}-{self.n_iters}'

    def make_player_str(self) -> str:
        player_args = {
            '--type': 'MCTS-C',
            '--name': self.__repr__(),
            '-i': self.n_iters,
            '-n': 1,
        }

        if self.gen == 0:
            player_args['--no-model'] = None
        else:
            player_args['-m'] = self.model_filename

        if self.set_temp_zero:
            player_args['--starting-move-temp'] = 0
            player_args['--ending-move-temp'] = 0

        return make_args_str(player_args)

    @property
    def version(self):
        return self.gen

@dataclass(frozen=True)
class ReferenceAgent(Agent):
    type_str: str
    strength_param: str
    strength: int
    binary_filename: str = None

    def __repr__(self):
        return f'{self.type_str}-{self.strength}'

    def make_player_str(self) -> str:
        player_args = {
            '--type': self.type_str,
            '--name': self.__repr__(),
            f'--{self.strength_param}': self.strength,
        }
        return make_args_str(player_args)

    @property
    def version(self):
        return self.strength