from dataclasses import dataclass
from enum import Enum
from typing import Dict
from util.torch_util import Shape


@dataclass
class ShapeInfo:
    name: str
    target_index: int
    primary: bool
    shape: Shape


class SearchParadigm(Enum):
    AlphaZero = 'alpha0'
    BetaZero = 'beta0'

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in {paradigm.value for paradigm in SearchParadigm}


ShapeInfoDict = Dict[str, ShapeInfo]
