from util.torch_util import Shape

import torch

from dataclasses import dataclass
from enum import Enum
from typing import Dict


@dataclass
class ShapeInfo:
    name: str
    target_index: int
    shape: Shape


ShapeInfoDict = Dict[str, ShapeInfo]


@dataclass
class ShapeInfoCollection:
    input_shapes: ShapeInfoDict
    target_shapes: ShapeInfoDict
    head_shapes: ShapeInfoDict


class SearchParadigm(Enum):
    AlphaZero = 'alpha0'
    BetaZero = 'beta0'
    GammaZero = 'gamma0'

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in {paradigm.value for paradigm in SearchParadigm}


HeadValuesDict = Dict[str, torch.Tensor]
