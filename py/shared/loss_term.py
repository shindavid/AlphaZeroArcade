from __future__ import annotations

from shared.basic_types import HeadValuesDict

import torch
from torch import nn as nn

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


class ValueUncertaintyLossTerm(LossTerm):
    """
    A LossTerm for a ValueUncertaintyHead.

    The ValueUncertaintyHead predicts D^2, where D is the difference between the Q_posterior target
    and the value head's output (converted from logit space to probability space). Technically, in a
    p-player game, D should be a length-p vector. If the game is zero-sum, then the space of
    possible D vectors is only (p-1)-dimensional. For now, ValueUncertaintyLossTerm assumes the game
    is 2-player zero-sum, so it predicts D as a scalar.

    Following KataGo, the loss term is computed as the Huber loss between the predicted D^2 and the
    actual D^2. This is quadratic in D; Huber is used rather than MSE to reduce sensitivity to
    outliers.
    """
    def __init__(self, name: str, value_name: str, Q_posterior_target_name: str, weight: float):
        """
        Args:
            name: the name of the ValueUncertaintyHead module
            value_name: the name of the head/target that provides the value prediction
            Q_posterior_target_name: the name of the head that provides the target Q posterior
            weight: the weight to assign to this loss term
        """
        super().__init__(name, weight)
        self._value_name = value_name
        self._Q_posterior_target_name = Q_posterior_target_name
        self._value_head = None  # initialized lazily in post_init()
        self._loss_fn = nn.HuberLoss()

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)

    def compute_loss(self, y: HeadValuesDict, y_hat: HeadValuesDict) -> torch.Tensor:
        Q_prior = self._value_head.to_win_share(y[self._value_name])  # (B, 2)
        Q_posterior = y_hat[self._Q_posterior_target_name]  # (B, 2)
        D = Q_posterior - Q_prior  # (B, 2)
        D1 = D[:, :1]  # (B, 1)
        Dsq = D1 * D1  # (B, 1)
        return self._loss_fn(y[self.name], Dsq)
