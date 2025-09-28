from __future__ import annotations

from shared.basic_types import HeadValuesDict

import torch

import abc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from shared.model import Model


class LossTerm:
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def post_init(self, model: Model):
        """
        Called after the Model has been constructed, to allow the LossTerm to initialize any
        necessary state based on the model.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, y: HeadValuesDict, y_hat: HeadValuesDict) -> torch.Tensor:
        pass


class BasicLossTerm(LossTerm):
    """
    BasicLossTerm assumes that head is the name of a module in the ModuleConfig, and that the module
    that it maps to is of type Head.

    Looks up that Head type, and invokes Head.default_loss_function() to get the loss function to
    use.
    """
    def __init__(self, head: str, weight: float):
        super().__init__(head, weight)
        self._loss_fn = None  # initialized lazily in post_init()

    def post_init(self, model: Model):
        self._loss_fn = model.get_head(self.name).default_loss_function()

    def compute_loss(self, y: HeadValuesDict, y_hat: HeadValuesDict) -> torch.Tensor:
        return self._loss_fn(y[self.name], y_hat[self.name])
