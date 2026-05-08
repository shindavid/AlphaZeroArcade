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
                 input_deps: Optional[Dict[str, FrozenSet[str]]] = None,
                 input_value_dict: Optional[Dict[str, torch.Tensor]] = None):
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
        input_value_dict: external-input name -> tensor of values fed into the model. Used by
                          loss terms that need to anchor a model output toward one of its own
                          inputs (e.g. BackupLossTerm's bootstrap-against-Q*/W* mechanism).
        """
        self.mask_dict = mask_dict
        self.y_hat_dict = y_hat_dict
        self.y_dict = y_dict
        self.input_mask_dict = input_mask_dict or {}
        self.input_deps = input_deps or {}
        self.input_value_dict = input_value_dict or {}

    def get_input(self, name: str, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the tensor that was fed as external input `name`. If `mask` is provided, the
        tensor is restricted to samples where mask is True (matching the convention of
        get_y_hat_and_y()).
        """
        t = self.input_value_dict[name]
        if mask is None:
            return t
        return apply_mask(t, mask)

    def compute_combined_mask(self, y_hat_names: List[str], y_names: List[str]) -> torch.Tensor:
        """
        Return the same per-sample mask used internally by get_y_hat_and_y() for the given
        y_hat / y selection: intersection of all relevant target masks and (transitively
        consumed) input masks.
        """
        masks: List[torch.Tensor] = [self.mask_dict[name] for name in y_names]
        for y_hat_name in y_hat_names:
            for input_name in self.input_deps.get(y_hat_name, ()):
                m = self.input_mask_dict.get(input_name)
                if m is not None:
                    masks.append(m)
        if not masks:
            raise ValueError('compute_combined_mask requires at least one mask source')
        mask = masks[0].clone()
        for m in masks[1:]:
            mask = mask & m
        return mask

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

    def set_generation(self, gen: int):
        """
        Called once per training epoch (before any compute_loss() calls) to inform the term of
        the current generation. Default is a no-op; loss terms with generation-dependent
        behavior (e.g. BackupLossTerm's bootstrap-anneal) override this.
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

    Bootstrap-anneal (see docs/BetaZero.pdf):
      Early in training we want the BackupNet to behave like AlphaZero's children-average
      aggregation, i.e. its outputs Q, W should match the LoTE baseline inputs Qs_star,
      Ws_star. We achieve this by blending the per-sample loss values (Option A):

          q_loss = alpha * q_loss_against_Q_star + (1 - alpha) * q_loss_against_true
          w_loss = alpha * w_loss_against_W_star + (1 - alpha) * w_loss_against_true

      where alpha = 1 at gen <= bootstrap_start_gen, alpha = 0 at gen >= bootstrap_end_gen,
      and linearly interpolated in between. The Q*-anchor regresses
      to_win_share(Q_logits)[:, 0] toward Qs_star (Huber, delta=0.1, matching the
      uncertainty-head convention); the W*-anchor regresses W toward Ws_star with the W
      head's own loss function. At alpha = 1 the only training signal is "match Qs_star /
      Ws_star exactly," which mimics AlphaZero. The default schedule is intentionally absurd
      (bootstrap_end_gen = 10**8) so v1 stays in pure-bootstrap mode -- a useful sanity
      check that BetaZero closely matches AlphaZero.
    """
    def __init__(self, name: str, weight: float,
                 q_weight: float, w_weight: float,
                 value_name: str = 'value',
                 value_uncertainty_name: str = 'value_uncertainty',
                 future_mcts_value_name: str = 'future_mcts_value',
                 qs_star_input_name: str = 'Qs_star',
                 ws_star_input_name: str = 'Ws_star',
                 bootstrap_start_gen: int = 0,
                 bootstrap_end_gen: int = 10 ** 8):
        super().__init__(name, weight)
        assert bootstrap_start_gen <= bootstrap_end_gen, (
            bootstrap_start_gen, bootstrap_end_gen)
        self._value_name = value_name
        self._value_uncertainty_name = value_uncertainty_name
        self._future_mcts_value_name = future_mcts_value_name
        self._qs_star_input_name = qs_star_input_name
        self._ws_star_input_name = ws_star_input_name
        self._bootstrap_start_gen = bootstrap_start_gen
        self._bootstrap_end_gen = bootstrap_end_gen
        self._alpha = 1.0  # default: full bootstrap until set_generation() is called
        self._q_weight = q_weight
        self._w_weight = w_weight
        self._value_head = None       # initialized lazily in post_init()
        self._q_loss_fn = None
        self._w_loss_fn = None
        # Q*-anchor loss; matches the Huber convention used by the uncertainty heads, so the
        # numerical scale of the bootstrap loss is comparable to the non-bootstrap loss.
        self._q_anchor_loss_fn = nn.HuberLoss(delta=0.1)

    def post_init(self, model: Model):
        self._value_head = model.get_head(self._value_name)
        vu_head = model.get_head(self._value_uncertainty_name)
        self._q_loss_fn = self._value_head.default_loss_function()()
        self._w_loss_fn = vu_head.default_loss_function()()

    def set_generation(self, gen: int):
        start = self._bootstrap_start_gen
        end = self._bootstrap_end_gen
        if gen <= start:
            self._alpha = 1.0
        elif gen >= end:
            self._alpha = 0.0
        else:
            # Linear interpolation: alpha=1 at start, alpha=0 at end.
            self._alpha = 1.0 - (gen - start) / float(end - start)

    def compute_loss(self, masker: Masker) -> Tuple[torch.Tensor, int]:
        y_hat_names = [self.name]
        y_names = [self._value_name, self._future_mcts_value_name]
        y_hat, y = masker.get_y_hat_and_y(y_hat_names, y_names)

        backup_out = y_hat[0]                         # (B, value_dim + 1)
        Q_logits = backup_out[:, :-1]                 # (B, value_dim)
        W_pred = backup_out[:, -1]                    # (B,)

        value_target = y[0]                           # (B, value_dim) probs/one-hot
        future_mcts_value = y[1]                      # (B, kNumPlayers) win-shares

        alpha = self._alpha

        # --- "true" targets path (alpha=0 limit) ---
        q_loss_true = self._q_loss_fn(Q_logits, value_target)

        Q_winshare = self._value_head.to_win_share(Q_logits.detach())  # (B, kNumPlayers)
        active_future = future_mcts_value[:, 0]                        # (B,)
        active_Q = Q_winshare[:, 0]                                    # (B,)
        w_target_true = (active_future - active_Q).square() + 1e-8
        w_loss_true = self._w_loss_fn(W_pred, w_target_true)

        # --- bootstrap-anchor path (alpha=1 limit): match Qs_star / Ws_star inputs ---
        if alpha > 0.0:
            mask = masker.compute_combined_mask(y_hat_names, y_names)
            qs_star = masker.get_input(self._qs_star_input_name, mask)  # (B', 1) or (B',)
            ws_star = masker.get_input(self._ws_star_input_name, mask)  # (B', 1) or (B',)
            qs_star = qs_star.reshape(-1)
            ws_star = ws_star.reshape(-1)

            # Pull active-seat win-share from full-grad Q (no detach: this is the path that
            # actually trains Q) toward the LoTE baseline Qs_star.
            active_Q_grad = self._value_head.to_win_share(Q_logits)[:, 0]   # (B',)
            q_loss_anchor = self._q_anchor_loss_fn(active_Q_grad, qs_star)

            w_target_anchor = ws_star + 1e-8
            w_loss_anchor = self._w_loss_fn(W_pred, w_target_anchor)

            q_loss = alpha * q_loss_anchor + (1.0 - alpha) * q_loss_true
            w_loss = alpha * w_loss_anchor + (1.0 - alpha) * w_loss_true
        else:
            q_loss = q_loss_true
            w_loss = w_loss_true

        loss = self._q_weight * q_loss + self._w_weight * w_loss
        return loss, len(W_pred)
