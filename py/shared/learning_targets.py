"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import abc
import math
from typing import Optional

import torch
from torch import nn as nn


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


class ActionValueTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

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

    def get_num_correct_predictions(self, predictions: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        predictions_max = torch.argmax(predictions, dim=1)
        labels_max = torch.argmax(labels, dim=1)
        return int(sum(predictions_max == labels_max))


class ValueTarget(LearningTarget):
    def loss_fn(self) -> nn.Module:
        """
        AlphaGo, KataGo, etc., use the standard [-1, +1] range for value targets, and thus use MSE
        loss. In our project, however, in an effort to generalize to M-player games, we use a
        length-M vector of non-negative values summing to 1. So for us cross-entropy loss is more
        appropriate.

        NOTE: in conversations with David Wu, he suggested relaxing the fixed-utility assumption,
        to support games like Backgammon where the utility is not fixed. In that case, we would need
        to use MSE loss. There are some subtle c++ changes that would need to be made to support
        this.
        """
        return nn.CrossEntropyLoss()

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        """
        Naively using the same implementation as PolicyTarget.get_num_correct_predictions() doesn't
        work for games that have draws. For example, in TicTacToe, if the true value is [0.5, 0.5],
        and the network outputs [0.4, 0.6], then the network should get credit for a correct
        prediction. But the naive implementation would give it no credit.

        Instead, in this example, we consider the output of [0.4, 0.6] to be correct, by virtue of
        the fact that abs([0.4, 0.6] - [0.5, 0.5]) = [0.1, 0.1] is "close enough" to [0, 0].
        """
        predicted_probs = predicted_logits.softmax(dim=1)
        deltas = abs(predicted_probs - labels)
        return int(sum((deltas < 0.25).all(dim=1)))


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
