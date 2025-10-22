from __future__ import annotations

from shared.net_modules import Head
from util.torch_util import apply_mask

import torch
from torch import nn as nn

import abc
import logging
from typing import List, Optional, TYPE_CHECKING, Tuple


if TYPE_CHECKING:
    from shared.model import Model


logger = logging.getLogger(__name__)


class Masker:
    def __init__(self, mask_dict, y_hat_dict, y_dict):
        self.mask_dict = mask_dict
        self.y_hat_dict = y_hat_dict
        self.y_dict = y_dict

    def get_y_hat_and_y(self, y_hat_names: List[str], y_names: List[str]):
        y_hats = [self.y_hat_dict[name] for name in y_hat_names]
        ys = [self.y_dict[name] for name in y_names]
        if not y_names:
            return y_hats, ys

        masks = [self.mask_dict[name] for name in y_names]
        mask = masks[0].clone()
        for m in masks[1:]:
            mask = mask & m

        y_hats = [apply_mask(y_hat, mask) for y_hat in y_hats]
        ys = [apply_mask(y, mask) for y in ys]
        return y_hats, ys


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
    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        """
        Return the loss, and the number of samples that contributed to the loss.
        """
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
        self._head: Optional[Head] = None
        self._use_policy_scaling = False
        self._loss_fn = None  # initialized lazily in post_init()

    def post_init(self, model: Model):
        self._head = model.get_head(self.name)
        self._use_policy_scaling = self._head.requires_policy_scaling()
        loss_fn_type = self._head.default_loss_function()
        if self._use_policy_scaling:
            self._loss_fn = loss_fn_type(reduction='none')
        else:
            self._loss_fn = loss_fn_type()

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name]
        y_names = [self.name]

        if self._use_policy_scaling:
            y_names.append('valid_actions')

        y_hat_list, y_list = masker.get_y_hat_and_y(y_hat_names, y_names)
        y_hat = y_hat_list[0]
        y = y_list[0]
        loss = self._loss_fn(y_hat, y)

        if self._use_policy_scaling:
            assert torch.isfinite(y_hat).all(), y_hat
            assert torch.isfinite(y).all(), y
            assert (y >= 0).all(), y
            # loss has not been reduced yet
            valid_actions = y_list[1]
            denominator = valid_actions.sum()
            if denominator == 0:
                loss = y_hat.sum() * 0.0
            else:
                while valid_actions.dim() < loss.dim():
                    valid_actions = valid_actions.unsqueeze(-1)

                unreduced_loss = loss * valid_actions      # mask out invalid actions
                loss = unreduced_loss.sum() / denominator  # reduce here

        return loss, len(y_hat)


class ValueUncertaintyLossTerm(LossTerm):
    """
    A LossTerm for a ValueUncertaintyHead.

    The ValueUncertaintyHead predicts a size-P tensor, T, where P is the number of players. For the
    p'th player, let Q_min(p) and Q_max(p) be the minimum and maximum values of Q ever observed for
    that player during MCTS search. Then T(p) corresponds to the larger of (V(p) - Q_min(p))^2 and
    (Q_max(p) - V(p))^2, where V(p) is the value head's output (converted from logit space to
    probability space).

    Following KataGo, the loss term is computed using Huber loss to reduce sensitivity to outliers.
    """
    def __init__(self, name: str, value_name: str, Q_min_target_name: str, Q_max_target_name: str,
                 weight: float):
        """
        Args:
            name: the name of the ValueUncertaintyHead module
            value_name: the name of the head/target that provides the value prediction
            Q_min_target_name: the name of the head that provides the target Q_min
            Q_max_target_name: the name of the head that provides the target Q_max
            weight: the weight to assign to this loss term
        """
        super().__init__(name, weight)
        self._value_name = value_name
        self._Q_min_target_name = Q_min_target_name
        self._Q_max_target_name = Q_max_target_name
        self._value_head = None  # initialized lazily in post_init()
        self._loss_fn = nn.HuberLoss()

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name, self._value_name]
        y_names = [self._Q_min_target_name, self._Q_max_target_name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        predicted_Dsq, win_value = y_hat
        Q_min = y[0]  # (B, 2)
        Q_max = y[1]  # (B, 2)

        Q_prior = self._value_head.to_win_share(win_value)  # (B, 2)
        actual_Dsq = torch.max((Q_prior - Q_min) ** 2, (Q_max - Q_prior) ** 2)  # (B, 2)
        return self._loss_fn(predicted_Dsq, actual_Dsq), len(predicted_Dsq)
