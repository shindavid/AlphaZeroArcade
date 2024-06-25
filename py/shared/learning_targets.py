"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import abc
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
    def convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Converts the labels produced by the c++ code into the format expected by the loss function.
        By default, this is a no-op; derived classes can override this.
        """
        return labels

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


class ScoreMarginTarget(LearningTarget):
    def __init__(self, max_score_margin: int, min_score_margin: Optional[int]=None):
        """
        min_score_margin defaults to -max_score_margin
        """
        self.max_score_margin = max_score_margin
        self.min_score_margin = -max_score_margin if min_score_margin is None else min_score_margin

    def convert_labels(self, categories: torch.Tensor) -> torch.Tensor:
        # categories -> one-hot
        assert len(categories.shape) == 2 and categories.shape[1]==1, categories.shape
        n = categories.shape[0]
        one_hot = torch.zeros((n, self.max_score_margin - self.min_score_margin + 1))
        index = categories[:, 0] - self.min_score_margin
        one_hot[torch.arange(n), index.type(torch.int64)] = 1
        return one_hot

    def loss_fn(self) -> nn.Module:
        """
        For the loss, we have a pdf component and a cdf component.

        For the pdf component, we use cross-entropy loss.

        For the cdf component, we use MSE loss.

        The arguments to the loss function are in pdf form, so the loss function has to do the work
        of producting the cdf from the pdf.
        """
        pdf_loss = nn.CrossEntropyLoss()
        cdf_loss = nn.MSELoss()

        def loss(predicted_logits: torch.Tensor, actual_one_hot: torch.Tensor):
            predicted_probs = predicted_logits.softmax(dim=1)
            predicted_cdf = torch.cumsum(predicted_probs, dim=1)
            actual_cdf = torch.cumsum(actual_one_hot, dim=1)

            pdf_loss_val = pdf_loss(predicted_logits, actual_one_hot)
            cdf_loss_val = cdf_loss(predicted_cdf, actual_cdf)

            return pdf_loss_val + cdf_loss_val

        return loss

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    actual_one_hot: torch.Tensor) -> float:
        return torch.sum(predicted_logits.softmax(dim=1) * actual_one_hot).item()


class OwnershipTarget(LearningTarget):
    def convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.type(torch.int64)  # needed for cross-entropy loss

    def loss_fn(self) -> nn.Module:
        """
        The tensors passed into this loss function are expected to be of shape (N, K, *board),
        where:

        N = batch size
        K = number of possible owners
        *board = the shape of the board

        For our loss function, we do a cross-entropy loss for each square, and then sum them up.

        nn.CrossEntropyLoss nicely handles this exact type of input!
        """
        return nn.CrossEntropyLoss()

    def get_num_correct_predictions(self, predicted_logits: torch.Tensor,
                                    actual_categories: torch.Tensor) -> float:
        n = predicted_logits.shape[0]
        predicted_categories = torch.argmax(predicted_logits, dim=1)
        matches = (predicted_categories == actual_categories).float()
        return matches.mean().item() * n
