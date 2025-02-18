from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class MCTSAgent:
    gen: int = 0
    n_iters: int = 0

    def __repr__(self):
        return f'{self.gen}-{self.n_iters}'

@dataclass(frozen=True)
class PerfectAgent:
    strength: int = 0
    def __repr__(self):
        return f'Perfect-{self.strength}'

@dataclass(frozen=True)
class RandomAgent:
    def __repr__(self):
        return 'Random'

Agent = Union[MCTSAgent, PerfectAgent, RandomAgent]