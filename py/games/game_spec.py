import abc
from dataclasses import dataclass
from typing import Dict, List, Optional

from net_modules import ModelConfigGenerator


@dataclass
class ReferencePlayerFamily:
    """
    A given game can have a reference player family, used to measure the skill level of the
    AlphaZero agent. Such a family is defined by a type string, passed as the --type argument of the
    --player command-line option for the game binary. The family should be parameterized by a
    single integer-valued parameter, which controls how well the agent plays.
    """
    type_str: str
    strength_param: str
    min_strength: int
    max_strength: int


class GameSpec(abc.ABC):
    """
    Abstract class for defining a game that can be played by the AlphaZero.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the game. This is used both to reference the game when running scripts, and as
        the name of the game binary/shared-library.

        The build process is expected to produce the following files:

        target/Release/bin/{name}
        target/Release/lib/lib{name}.so
        """
        pass

    @property
    def extra_runtime_deps(self) -> List[str]:
        """
        List of extra files that are required to run the game, related to the repo root.

        These files will be copied to the target/Release/bin/extra/ directory by the build process,
        and to the alphazero dir's bins/extra/ directory by the alphazero process.
        """
        return []

    @property
    @abc.abstractmethod
    def model_configs(self) -> Dict[str, ModelConfigGenerator]:
        """
        Dictionary of model configurations for this game.

        The keys are model names, and the values are ModelConfigGenerators.
        """
        return {}

    @property
    def reference_player_family(self) -> Optional[ReferencePlayerFamily]:
        """
        The reference player family for this game, if any.
        """
        return None

    @property
    def training_options(self) -> Dict[str, str]:
        """
        Options to pass to the game binary when running training games.
        """
        return {}

    @property
    def training_player_options(self) -> Dict[str, str]:
        """
        Options to pass to the --player argument of the game binary when running training games.
        """
        return {}

    @property
    def rating_options(self) -> Dict[str, str]:
        """
        Options to pass to the game binary when running ratings games.
        """
        return {}

    @property
    def rating_player_options(self) -> Dict[str, str]:
        """
        Options to pass to the --player argument of the game binary when running ratings games.
        """
        return {}
