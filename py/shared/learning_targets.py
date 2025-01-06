"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import abc
import math
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F


class LearningTarget:
    """
    Each model head has a LearningTarget, which bundles a number of useful pieces of information
    together:

    - The loss function to use for this head

    (that's it...should we remove this class and just specify the loss function directly?)
    """
    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass


class PolicyTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class WinShareActionValueTarget(LearningTarget):
    """
    WinShareActionValueTarget is a LearningTarget for the action-value head for games where the
    MCTS-propagated value is a win-share value. This is the case for both
    WinShareDrawActionValueHead and WinLossDrawActionValueHead.
    """
    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()


class WinLossDrawValueTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class KLDivergencePerPixelLoss(nn.Module):
    """
    KL-divergence loss computed per "pixel", averaged over all pixels.

    Assumes 1D-distributions per-pixel, and an arbitrary shape for the pixel dimensions. This is
    similar to how CrossEntropyLoss works permits pixel-wise classification.

    This is currently unused, and is buggy anyways. Maybe I'll bring it back at some point.
    """
    def __init__(self):
        super(KLDivergencePerPixelLoss, self).__init__()

    def forward(self, logits, target):
        """
        Args:
            logits: Tensor of shape (N, C, *) where N is the batch size, C is the number of classes,
                    and * represents any number of additional dimensions for "pixels".
            target: Tensor of shape (N, C, *) where each pixel is a valid probability distribution.

        Returns:
            Mean KL-divergence loss over all "pixels".
        """
        # Ensure logits and target have the same shape
        assert logits.shape == target.shape, "Logits and target must have the same shape."

        # Apply log-softmax to the logits to get log probabilities
        log_prob = F.log_softmax(logits, dim=1)  # Log probabilities along class dim (C)

        # Compute KL-divergence: target * (log(target) - log(predicted))
        kl_div = target * (torch.log(target + 1e-10) - log_prob)  # Small epsilon to avoid log(0)

        # Sum over the class dimension (C) to get the KL-divergence for each "pixel"
        kl_div = torch.sum(kl_div, dim=1)  # Sum over the class dimension C

        # Return the mean KL-divergence over all "pixels" and batches
        return kl_div.mean()


class WinShareValueTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        # TODO: try KLDivLoss instead of CrossEntropyLoss
        # return nn.KLDivLoss(reduction='batchmean')
        return nn.CrossEntropyLoss()


class ScoreTarget(LearningTarget):
    """
    Tensors are of shape (B, D, C, aux...), where:

    B = batch-size
    D = num distribution types (2: PDF, CDF)
    C = number of classes (i.e., number of possible scores)
    aux... = any number of additional dimensions

    aux... might be nonempty if for example we're predicting for multiple players
    """
    def loss_fn(self) -> nn.Module:
        pdf_loss_fn = nn.CrossEntropyLoss()
        cdf_loss_fn = nn.MSELoss()

        def loss(output: torch.Tensor, target: torch.Tensor):
            assert output.shape == target.shape, (output.shape, target.shape)
            assert output.shape[1] == 2, output.shape  # PDF, CDF

            pdf_loss = pdf_loss_fn(output[:, 0], target[:, 0])
            cdf_loss = cdf_loss_fn(output[:, 1], target[:, 1])

            return pdf_loss + cdf_loss

        return loss


class OwnershipTarget(LearningTarget):
    """
    Tensors are of shape (B, C, aux...), where:

    B = batch-size
    C = number of classes (i.e., number of possible ownership categories)
    aux... = any number of additional dimensions

    aux... will typically be the board dimensions.
    """
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class GeneralLogitTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()
