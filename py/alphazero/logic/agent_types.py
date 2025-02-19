from util.str_util import make_args_str

from dataclasses import dataclass
from abc import ABC, abstractmethod
import os


class Agent(ABC):
    """
    Base class for agents. All agents must implment make_player_str() for generating
    command-line arguments for the player.
    """
    @abstractmethod
    def make_player_str(self):
        pass


@dataclass(frozen=True)
class MCTSAgent(Agent):
    gen: int = 0
    n_iters: int = 0
    model_dir: str = None

    def __repr__(self):
        return f'{self.gen}-{self.n_iters}'

    def make_player_str(self, set_temp_zero: bool=False) -> str:
        player_args = {
            '--type': 'MCTS-C',
            '--name': self.__repr__(),
            '-i': self.n_iters,
            '-m': self.get_model_path(),
            '-n': 1,
        }

        if set_temp_zero:
            player_args['--starting-move-temp'] = 0
            player_args['--ending-move-temp'] = 0

        return make_args_str(player_args)

    def get_model_path(self) -> str:
        return os.path.join(self.model_dir, f'gen-{self.gen}.pt')

@dataclass(frozen=True)
class PerfectAgent(Agent):
    strength: int = 0
    def __repr__(self):
        return f'Perfect-{self.strength}'

    def make_player_str(self, set_temp_zero: bool=False) -> str:
        strength = self.strength
        player_args = {
            '--type': 'Perfect',
            '--name': self.__repr__(),
            '--strength': strength,
        }
        return make_args_str(player_args)

@dataclass(frozen=True)
class UniformAgent(Agent):
    n_iters: int = 0

    def __repr__(self):
        return f'Uniform-{self.n_iters}'

    def make_player_str(self, set_temp_zero: bool=False) -> str:
        player_args = {
            '--type': 'MCTS-C',
            '--name': self.__repr__(),
            '-i': self.n_iters,
            '-n': 1,
            '--no-model': None,
        }
        return make_args_str(player_args)
