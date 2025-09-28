from util.torch_util import Shape

import torch

from dataclasses import dataclass
from enum import Enum
from typing import Dict


@dataclass
class ShapeInfo:
    name: str
    target_index: int
    primary: bool
    shape: Shape


ShapeInfoDict = Dict[str, ShapeInfo]


class SearchParadigm(Enum):
    AlphaZero = 'alpha0'
    BetaZero = 'beta0'

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in {paradigm.value for paradigm in SearchParadigm}


HeadValuesDict = Dict[str, torch.Tensor]
