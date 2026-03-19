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
        y_hat_list, y_list = masker.get_y_hat_and_y(y_hat_names, y_names)
        y_hat = y_hat_list[0]
        y = y_list[0]

        if self._use_policy_scaling:
            if not torch.isfinite(y_hat).all() or not torch.isfinite(y).all():
                # print the parts of y_hat and y that are not finite for debugging
                dbg_pairs = [('y_hat', y_hat), ('y', y)]
                for name, tensor in dbg_pairs:
                    if torch.isfinite(tensor).all():
                        continue

                    # print the first 10 rows of the tensor that contain non-finite values, along with the non-finite values themselves:
                    non_finite_mask = ~torch.isfinite(tensor)
                    non_finite_rows = non_finite_mask.any(dim=1).nonzero(as_tuple=False).squeeze()[:10]
                    logger.error('bad %s: %s', name, tensor[non_finite_rows])
                    logger.error('bad %s non-finite values: %s', name, tensor[non_finite_mask])
                raise ValueError('Non-finite loss term: %s' % self.name)

            mask = (y >= 0)
            denominator = mask.flatten(start_dim=2).any(dim=2).sum()

            if denominator == 0:
                loss = y_hat.sum() * 0.0
            else:
                y[~mask] = 0.0  # set invalid targets to 0 so they don't lead to nan's
                loss = self._loss_fn(y_hat, y)
                unreduced_loss = loss * mask      # mask out invalid actions
                loss = unreduced_loss.sum() / denominator  # reduce here
        else:
            loss = self._loss_fn(y_hat, y)

        return loss, len(y_hat)


class ValueUncertaintyLossTerm(LossTerm):
    """
    A LossTerm for a ValueUncertaintyHead.

    For the p'th player, define the following:

    - Q_min(p): minimum value of Q ever observed for that player during MCTS search
    - Q_max(p): maximum value of Q ever observed for that player during MCTS search
    - W(p): final W observed for that player during MCTS search

    Then, let T(p) be the largest of the following 3 quantities:

    1. (V(p) - Q_min(p))^2
    2. (Q_max(p) - V(p))^2
    3. W(p)

    The ValueUncertaintyHead predicts T.

    Following KataGo, the loss term is computed using Huber loss to reduce sensitivity to outliers.
    """
    def __init__(self, name: str, weight: float, value_name: str='value',
                 Q_min_target_name: str='Q_min', Q_max_target_name: str='Q_max',
                 W_target_name: str='W'):
        """
        Args:

        - name: the name of the ValueUncertaintyHead module
        - value_name: the name of the head/target that provides the value prediction
        - Q_min_target_name: the name of the head that provides the target Q_min
        - Q_max_target_name: the name of the head that provides the target Q_max
        - W_target_name: the name of the head that provides the target W
        - weight: the weight to assign to this loss term
        """
        super().__init__(name, weight)
        self._value_name = value_name
        self._Q_min_target_name = Q_min_target_name
        self._Q_max_target_name = Q_max_target_name
        self._W_target_name = W_target_name
        self._value_head = None  # initialized lazily in post_init()
        self._loss_fn = nn.MSELoss()

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name, self._value_name]
        y_names = [self._Q_min_target_name, self._Q_max_target_name, self._W_target_name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        U01, lR = y_hat
        lR = lR.detach()  # (B, 3)
        Q_min = y[0]  # (B, 2)
        Q_max = y[1]  # (B, 2)
        W = y[2]      # (B, 2)

        V = self._value_head.to_win_share(lR)  # (B, 2)

        d1 = (V - Q_min).square()
        d2 = (Q_max - V).square()
        d3 = W
        d_max = torch.max(torch.max(d1, d2), d3)  # (B, 2)
        d_cap = V * (1 - V)
        actual = torch.min(d_cap, d_max)  # cap at maximum possible variance

        U = U01 * d_cap
        loss = self._loss_fn(U, actual)
        if not torch.isfinite(loss).all():
            dbg_pairs = [
                ('U01', U01),
                ('U', U),
                ('actual', actual),
                ('lR', lR),
                ('V', V),
                ('Q_min', Q_min),
                ('Q_max', Q_max),
                ('W', W)
            ]
            for name, tensor in dbg_pairs:
                if not torch.isfinite(tensor).all():
                    logger.error('bad %s: %s', name, tensor[~torch.isfinite(tensor).all(dim=1)])
            logger.error('loss: %s', loss)
            raise ValueError('Non-finite U loss')
        return loss, len(U)


class ActionValueUncertaintyLossTerm(LossTerm):
    def __init__(self, name: str, weight: float, action_value_name: str='action_value'):
        super().__init__(name, weight)
        self._action_value_name = action_value_name
        self._action_value_head = None  # initialized lazily in post_init()
        self._loss_fn = nn.MSELoss(reduction='none')

    def post_init(self, model: Model):
        self._action_value_head = model.get_head(self._action_value_name)

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name, self._action_value_name]
        y_names = [self.name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        AU01_hat, lAV = y_hat
        lAV = lAV.detach()  # (B, A, 2)
        AV = torch.softmax(lAV, dim=2)  # (B, A, 2)
        AU = y[0]     # (B, A, 2)
        mask = (AU >= 0)
        denominator = mask.flatten(start_dim=2).any(dim=2).sum()

        d_cap = AV * (1 - AV)
        AU = torch.min(d_cap, AU)  # cap at maximum possible variance
        AU[~mask] = 0.5  # set invalid targets to 0.5 so they don't lead to nan's

        AU_hat = AU01_hat * d_cap

        if denominator == 0:
            loss = AU_hat.sum() * 0.0
        else:
            loss = self._loss_fn(AU_hat, AU)  # (B, A, 2)
            unreduced_loss = loss * mask
            loss = unreduced_loss.sum() / denominator  # reduce here

        if not torch.isfinite(loss).all():
            dbg_pairs = [
                ('AU01_hat', AU01_hat),
                ('AU_hat', AU_hat),
                ('AU', AU),
                ('AV', AV),
            ]
            for name, tensor in dbg_pairs:
                if not torch.isfinite(tensor).all():
                    logger.error('bad %s: %s', name, tensor[~torch.isfinite(tensor).all(dim=1)])
            logger.error('loss: %s', loss)
            raise ValueError('Non-finite AU loss')

        return loss, len(AU)
