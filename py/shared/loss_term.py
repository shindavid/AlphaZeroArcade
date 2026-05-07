from __future__ import annotations

from shared.net_modules import Head
from util.torch_util import apply_mask

import torch
from torch import nn as nn

import abc
import logging
from typing import Dict, FrozenSet, List, Optional, TYPE_CHECKING, Tuple


if TYPE_CHECKING:
    from shared.model import Model


logger = logging.getLogger(__name__)


class Masker:
    def __init__(self, mask_dict, y_hat_dict, y_dict,
                 input_mask_dict: Optional[Dict[str, torch.Tensor]] = None,
                 input_deps: Optional[Dict[str, FrozenSet[str]]] = None):
        """
        mask_dict:        target name -> per-sample bool mask (1=valid).
        y_hat_dict:       module name -> model output tensor.
        y_dict:           target name -> target tensor.
        input_mask_dict:  external-input name -> per-sample bool mask. External inputs that are
                          always valid (e.g. 'input') may be omitted.
        input_deps:       module name -> frozenset of external-input names the module's output
                          depends on (per the static DAG). When provided alongside
                          input_mask_dict, get_y_hat_and_y() additionally intersects the masks
                          of the inputs each requested y_hat depends on, ensuring loss is
                          computed only on samples where every consumed model output is
                          well-defined.
        """
        self.mask_dict = mask_dict
        self.y_hat_dict = y_hat_dict
        self.y_dict = y_dict
        self.input_mask_dict = input_mask_dict or {}
        self.input_deps = input_deps or {}

    def get_y_hat_and_y(self, y_hat_names: List[str], y_names: List[str]):
        y_hats = [self.y_hat_dict[name] for name in y_hat_names]
        ys = [self.y_dict[name] for name in y_names]

        # Collect all per-sample masks that must be intersected: target masks for the y's, plus
        # input masks for every external input transitively consumed by any requested y_hat.
        masks: List[torch.Tensor] = [self.mask_dict[name] for name in y_names]
        for y_hat_name in y_hat_names:
            for input_name in self.input_deps.get(y_hat_name, ()):
                m = self.input_mask_dict.get(input_name)
                if m is not None:
                    masks.append(m)

        if not masks:
            return y_hats, ys

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
    """
    def __init__(self, name: str, weight: float, value_name: str='value',
                 future_mcts_value_name: str='future_mcts_value'):
        super().__init__(name, weight)
        self._value_name = value_name
        self._future_mcts_value_name = future_mcts_value_name
        self._value_head = None  # initialized lazily in post_init()
        self._loss_fn = None  # initialized lazily in post_init()

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)
        head = model.get_head(self.name)
        self._loss_fn = head.default_loss_function()()

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name, self._value_name]
        y_names = [self._future_mcts_value_name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        predicted_sq_delta, lR = y_hat
        lR = lR.detach()  # (B, 3)
        V = self._value_head.to_win_share(lR)  # (B, 2)

        future_mcts_value = y[0]  # (B, 2)

        actual_sq_delta = (future_mcts_value - V).square()  # (B, 2)

        # Add small constant, matching KataGo.
        #
        # Claude explains: The point is to make the regression target slightly positive everywhere,
        # so the network has a non-trivial thing to predict on "perfect" positions instead of being
        # pulled toward exact zero (where softplus has near-zero gradient).
        actual_sq_delta += 1e-8
        loss = self._loss_fn(predicted_sq_delta, actual_sq_delta)
        return loss, len(predicted_sq_delta)


class BackupLossTerm(LossTerm):
    """
    A LossTerm for the BackupNet (BetaZero CPU-side NNUE).

    The BackupNet's output is a (B, value_dim + 1) tensor where:
      * columns [0 : value_dim] are Q logits in the same format as the base-NN's value head
        (e.g. WLD logits for WinLossDrawValueHead). These are trained against the same target
        and with the same loss function as the value head's own output.
      * column  [value_dim]      is W (uncertainty), the predicted squared error between the
        Q-derived active-seat win-share and the active-seat end-of-search MCTS value. This is
        trained against the same target and loss function as ValueUncertaintyHead.

    Restriction to backup-regime samples happens automatically via DAG-based input-mask
    intersection: the BackupNet depends on external inputs `Qs_star`, `Ws_star`, and
    `child_stats`, all of which are valid only on backup-regime samples.

    Calibration: q_weight and w_weight are intended to mirror the loss weights of the value
    and value_uncertainty heads, so that the Q vs. W training signal in the BackupNet matches
    the value vs. value_uncertainty signal in the base network.
    """
    def __init__(self, name: str, weight: float,
                 q_weight: float, w_weight: float,
                 value_name: str = 'value',
                 value_uncertainty_name: str = 'value_uncertainty',
                 future_mcts_value_name: str = 'future_mcts_value'):
        super().__init__(name, weight)
        self._value_name = value_name
        self._value_uncertainty_name = value_uncertainty_name
        self._future_mcts_value_name = future_mcts_value_name
        self._q_weight = q_weight
        self._w_weight = w_weight
        self._value_head = None       # initialized lazily in post_init()
        self._q_loss_fn = None
        self._w_loss_fn = None

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)
        vu_head = model.get_head(self._value_uncertainty_name)
        self._q_loss_fn = self._value_head.default_loss_function()()
        self._w_loss_fn = vu_head.default_loss_function()()

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name]
        y_names = [self._value_name, self._future_mcts_value_name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        backup_out = y_hat[0]                         # (B, value_dim + 1)
        Q_logits = backup_out[:, :-1]                 # (B, value_dim)
        W_pred = backup_out[:, -1]                    # (B,)

        value_target = y[0]                           # (B, value_dim) probs/one-hot
        future_mcts_value = y[1]                      # (B, kNumPlayers) win-shares

        # Q-loss: same loss function the value head uses against the same game-result target.
        q_loss = self._q_loss_fn(Q_logits, value_target)

        # W-loss: same loss function ValueUncertaintyHead uses, against the squared delta
        # between active-seat end-of-search value and the active-seat win-share derived from
        # Q. Detach Q so the W-loss does not pull Q toward future_mcts_value. Add 1e-8 to
        # keep the regression target strictly positive (matching ValueUncertaintyLossTerm).
        Q_winshare = self._value_head.to_win_share(Q_logits.detach())  # (B, kNumPlayers)
        active_future = future_mcts_value[:, 0]                        # (B,)
        active_Q = Q_winshare[:, 0]                                    # (B,)
        W_target = (active_future - active_Q).square() + 1e-8
        w_loss = self._w_loss_fn(W_pred, W_target)

        loss = self._q_weight * q_loss + self._w_weight * w_loss
        return loss, len(W_pred)
