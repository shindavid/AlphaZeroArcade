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
    - Whether or how to mask rows of data
    - An accuracy measurement function
    """
    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass

    def get_mask(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Assumes labels is a 2D tensor of shape (batch_size, num_labels). Returns a 1D tensor of
        shape (batch_size,)

        If no mask should be applied, return None. This is the default behavior; derived classes
        can override this.
        """
        return None

    @abc.abstractmethod
    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        pass


class PolicyTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        The C++ uses the zero-tensor to represent a row that should be masked.
        """
        assert len(labels.shape) == 2, labels.shape
        n = labels.shape[0]
        labels = labels.reshape((n, -1))

        label_sums = labels.sum(dim=1)
        mask = label_sums != 0
        return mask

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        selected_moves = torch.argmax(predicted_logits, dim=1)
        correct_policy_preds = labels.gather(1, selected_moves.view(-1, 1))
        return int(sum(correct_policy_preds))


class ActionValueTargetBase(LearningTarget):
    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        (B, C, A) = predicted_logits.shape  # batch-size, num-classes, num-actions
        assert predicted_logits.shape == labels.shape, (predicted_logits.shape, labels.shape)

        label_sums = labels.sum(dim=1)

        # check that all label_sums are close to 1:
        all_ok = torch.allclose(label_sums, torch.ones_like(label_sums))
        assert all_ok, labels

        predicted_probs = predicted_logits.softmax(dim=1)
        error = 0.5 * abs(predicted_probs - labels).sum().item()

        num_invalid_labels = labels[:, C - 1, :].sum().item()  # last class is invalid
        num_valid_labels = B * A - num_invalid_labels
        assert num_valid_labels > 0, num_valid_labels

        # To be extra strict, we exclude invalid labels from the denominator
        error_rate = error / num_valid_labels
        accuracy_rate = max(0, 1 - error_rate)
        return accuracy_rate * B


class ValueTargetBase(LearningTarget):
    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        (B, _) = predicted_logits.shape  # batch-size, num-classes
        assert predicted_logits.shape == labels.shape, (predicted_logits.shape, labels.shape)

        predicted_probs = predicted_logits.softmax(dim=1)
        error = 0.5 * abs(predicted_probs - labels).sum().item()

        return B - error


class WinLossDrawActionValueTarget(ActionValueTargetBase):
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class WinLossDrawValueTarget(ValueTargetBase):
    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class KLDivergencePerPixelLoss(nn.Module):
    """
    KL-divergence loss computed per "pixel", averaged over all pixels.

    Assumes 1D-distributions per-pixel, and an arbitrary shape for the pixel dimensions. This is
    similar to how CrossEntropyLoss works permits pixel-wise classification.
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


class WinShareActionValueTarget(ActionValueTargetBase):
    def loss_fn(self) -> nn.Module:
        # TODO: try KLDivergencePerPixelLoss instead of CrossEntropyLoss
        # return KLDivergencePerPixelLoss()
        return nn.CrossEntropyLoss()


class WinShareValueTarget(ValueTargetBase):
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

    def get_num_correct_predictions(self, output: torch.Tensor, target: torch.Tensor) -> float:
        return torch.sum(output[:, 0].softmax(dim=1) * target[:, 0]).item()


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

    def get_num_correct_predictions(self, output: torch.Tensor, target: torch.Tensor) -> float:
        shape = output.shape
        n = math.prod(shape[2:])
        return torch.sum(output.softmax(dim=1) * target).item() / n


class GeneralLogitTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def get_num_correct_predictions(self, prediction_logits: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        prediction_probs = prediction_logits.sigmoid()
        n = math.prod(labels.shape[1:])

        # If the label is 1, then a prediction of p counts as p correct predictions.
        # If the label is 0, then a prediction of p counts as 1-p correct predictions.
        w = labels * prediction_probs + (1 - labels) * (1 - prediction_probs)
        return torch.sum(w).item() / n
