from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.str_util import make_args_str

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    n_iters: int = 0
    set_temp_zero: bool = None
    tag: str = None

    def make_player_str(self, organizer: DirectoryOrganizer) -> str:
        assert organizer.tag == self.tag

        player_args = {
            '--type': 'MCTS-C',
            '--name': self.__repr__(),
            '-i': self.n_iters,
            '-n': 1,
        }

        if self.gen == 0:
            player_args['--no-model'] = None
        else:
            player_args['-m'] = organizer.get_model_filename(self.gen)

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
            '--name': self.__repr__(),
            f'{self.strength_param}': self.strength,
        }
        return make_args_str(player_args)
