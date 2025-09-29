"""
Most of the code in this file is based on classes defined in py/model_pytorch.py in the KataGo
codebase.

Note that KataGo has the requirement of supporting multiple board-sizes, and this in turn leads to
some additional complexity in the form of mask operations with additional carefully crated
normalization steps. We do not have this requirement, which has allowed for some simplifications.

Additionally, KataGo has sophisticated weight initialization and regularization schemes. For now,
we leave these out, relying on pytorch's defaults.

Note that as a whole, KataGo uses pre-activation residual blocks, while AlphaGo Zero uses
post-activation residual blocks. We follow KataGo and use pre-activation throughout.

KataGo paper: https://arxiv.org/pdf/1902.10565.pdf
AlphaGo Zero paper: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
"""
from shared.transformer_modules import TransformerBlock
from util.torch_util import LossFunction, Shape

import torch
from torch import nn as nn
from torch.nn import functional as F

import logging
import math
from typing import Optional


logger = logging.getLogger(__name__)


class Head(nn.Module):
    def default_loss_function(self) -> Optional[LossFunction]:
        return None


class GlobalPoolingLayer(nn.Module):
    """
    This corresponds to KataGPool in the KataGo codebase.

    The KataGo paper describes this layer as:

    1. The mean of each channel
    2. The mean of each channel multiplied by 1/10 ( b - b_avg )
    3. The maximum of each channel.

    https://arxiv.org/pdf/1902.10565.pdf

    Of these, the second one is only needed because KataGo uses mixed board-sizes. With a fixed
    board-size, the second one collapses to zero. In our context, then, we omit the second one.
    """
    NUM_CHANNELS = 2  # for KataGo this is 3, see above

    def __init__(self):
        super(GlobalPoolingLayer, self).__init__()

    def forward(self, x):
        g_mean = torch.mean(x, keepdim=True, dim=(2, 3))
        g_max, _ = torch.max(x.view(x.shape[:2] + (-1,)), dim=-1, keepdim=True)
        return torch.cat([g_mean, g_max[..., None]], dim=1)


class ConvBlock(nn.Module):
    """
    This corresponds to NormActConv with c_gpool=None in the KataGo codebase.

    The KataGo paper does not explicitly name this block, but its components are described like
    this:

    1. A batch-normalization layer.
    2. A ReLU activation function.
    3. A 3x3 convolution outputting c channels.

    For reference, the AlphaGo Zero paper describes the convolutional block as follows:

    1. A convolution of 256 filters of kernel size 3 x 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    """

    def __init__(self, c_in: int, c_out: int):
        super(ConvBlock, self).__init__()
        self.norm = nn.BatchNorm2d(c_in)
        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = x
        out = self.act(self.norm(out))
        out = self.conv(out)
        return out


class KataGoNeck(nn.Module):
    """
    A final batch-norm + activation layer, before the heads, as described in the KataGo paper.

    In my experiments, I found that this layer worsens network accuracy in c4. I don't have a good
    hypothesis for why this is the case.
    """
    def __init__(self, c_in: int):
        super(KataGoNeck, self).__init__()
        self.norm = nn.BatchNorm2d(c_in)
        self.act = F.relu

    def forward(self, x):
        out = x
        out = self.act(self.norm(out))
        return out


class ConvBlockWithGlobalPooling(nn.Module):
    """
    This corresponds to NormActConv with c_gpool!=None in the KataGo codebase. This has no
    analog in the AlphaGo Zero paper.

    The KataGo paper does not explicitly name this block, but it is described in terms of a
    "global pooling bias structure", which is described like this:

    - takes input tensors X (shape b x b x cX) and G (shape b x b x cG)
    - consists of:
        - A batch normalization layer and ReLU activation applied to G (output shape b x b x cG).
        - A global pooling layer (output shape 3cG).
        - A fully connected layer to cX outputs (output shape cX).
        - Channelwise addition with X, treating the cX values as per-channel biases (output shape
          b x b x cX)

    It should be noted that there are slight differences between the description in the paper and
    the implementation in the codebase. We follow the codebase here.

    The KataGo codebase has an intermediate class called KataConvAndGPool corresponding to this
    global pooling bias structure. Here, that class is effectively merged into this one.
    """
    def __init__(self, c_in: int, c_out: int, c_gpool: int):
        super(ConvBlockWithGlobalPooling, self).__init__()
        self.norm = nn.BatchNorm2d(c_in)
        self.norm_g = nn.BatchNorm2d(c_gpool)
        self.act = F.relu
        self.conv_r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.conv_g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.pool_g = GlobalPoolingLayer()
        self.linear_g = nn.Linear(GlobalPoolingLayer.NUM_CHANNELS * c_gpool, c_out, bias=False)

    def forward(self, x):
        out = x
        out = self.act(self.norm(out))

        out_r = self.conv_r(out)

        out_g = self.conv_g(out)
        out_g = self.act(self.norm_g(out_g))
        out_g = self.pool_g(out_g).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)

        out = out_r + out_g
        return out


class ResBlock(nn.Module):
    """
    This corresponds to ResBlock with c_gpool=None in the KataGo codebase. As in the KataGo
    codebase, we construct this by composing ConvBlock's. By contrast, the AlphaGo Zero paper
    describes this independently of their ConvBlock description.

    Both the KataGo paper and the AlphaGo Zero paper effectively describe this block as a
    composition of two ConvBlocks with a skip connection.
    """
    def __init__(self, c_in_out: int, c_mid: int):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(c_in_out, c_mid)
        self.conv2 = ConvBlock(c_mid, c_in_out)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return x + out


class ResBlockWithGlobalPooling(nn.Module):
    """
    This corresponds to ResBlock with c_gpool!=None in the KataGo codebase. This has no
    analog in the AlphaGo Zero paper.
    """
    def __init__(self, c_in_out: int, c_mid_total: int, c_mid_gp: int):
        super(ResBlockWithGlobalPooling, self).__init__()
        assert 0 < c_mid_gp < c_mid_total
        self.conv1 = ConvBlockWithGlobalPooling(c_in_out, c_mid_total - c_mid_gp, c_mid_gp)
        self.conv2 = ConvBlock(c_mid_total - c_mid_gp, c_in_out)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return x + out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, board_size, dropout=0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(board_size, d_model)
        position = torch.arange(0, board_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PolicyHead(Head):
    """
    This maps to the main subhead of PolicyHead in the KataGo codebase.

    Per David Wu's advice on our 2023-10-26 Zoom meeting, I am NOT replicating KataGo's head
    architecture here. Some notes on why:

    - KataGo incorporates a gpool layer, and he says that this only exists because of the need to
      predict the "pass" move, which is a special consideration in the game of go and not generally
      applicable to other games. For other games, he feels the gpool layer should not be needed.
      I asked about global strategic considerations like kos in go, and he says that the global
      layers in the trunk should propagate that information already.

    - KataGo forgoes a final linear layer in favor of a convolutional layer. This works because the
      spatial shape of the input matches the spatial shape of the policy output. This is not
      generally true in all games. In the future, I can allow for specialization in games that share
      this property by providing a convolution-based policy head class, or perhaps even by making
      this policy head class automatically default to a convolution when it detects a spatial shape
      match, but that is premature at this time.

    - He suggested that for a more robust general purpose policy head, we can use an attention
      layer. If we want to help the network along in games with input-vs-policy spatial match, we
      can achieve this by initializing the weights of the layer specially. This deserves some
      careful experimentation if we go this route.

    With that said, we do follow KataGo in the following respects:

    - pre-activation instead of post-activation
    - no batch-norms

    For reference, the AlphaGo Zero paper describes the policy head as follows:

    1. A convolution of 2 filters of kernel size 1 x 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19^2 + 1 = 362 corresponding to
    logit probabilities for all intersections and the pass move
    """
    def __init__(self, spatial_size: int, c_in: int, c_hidden: int, output_shape: Shape):
        super(PolicyHead, self).__init__()

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, math.prod(output_shape))

    def default_loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.output_shape)
        return out


class WinLossDrawValueHead(Head):
    """
    A head that produces a length-3 logit vector, usable for 2-player games that permit draws.

    This maps to the main subhead of ValueHead in the KataGo codebase.

    Per David Wu's advice on our 2023-10-26 Zoom meeting, I am NOT replicating KataGo's head
    architecture here. Some notes on why:

    - KataGo collapses the spatial information into a gpool layer, and predicts off of that. He says
      that this is actually taking advantage of special properties of the game of go: the gpool
      layer collapses the spatial channels into mean values, and the game of go happens to decide
      the winner based on comparing mean black ownership vs mean white ownership. If the rules of
      go were slightly tweaked, the gpool layer would no longer have the appropriate information.
      A general purpose value head should not rely on this. (Although, we could do the same for the
      game of Othello.)

    With that said, we do follow KataGo in the following respects:

    - pre-activation instead of post-activation
    - no batch-norms

    For reference, the AlphaGo Zero paper describes the value head as follows:

    1. A convolution of 1 filter of kernel size 1 x 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    def __init__(self, spatial_size: int, c_in: int, c_hidden: int, n_hidden: int):
        super(WinLossDrawValueHead, self).__init__()

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 3)

    def default_loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


class WinLossDrawValueUncertaintyHead(Head):
    """
    A head that produces a length-1 logit vector, usable for 2-player games that permit draws.

    This is an uncertainty head that can be used in conjunction with WinLossDrawValueHead to
    represent uncertainty in the value prediction. The output is interpreted as logit(delta^2),
    where delta is the difference between the value prediction (converted from W/L/D to win-shared)
    and the true value.

    See the "Uncertainty-Weighted MCTS Playouts" section here:

    https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
    """
    def __init__(self):
        # TODO: this head will take as input the output of a WinLossDrawValueHead, along with
        # the trunk output, and will combine them in some way before producing its final output.
        pass


class WinLossValueHead(Head):
    """
    A head that produces a length-2 logit vector, usable for 2-player games that do NOT permit
    draws.

    This is based off WinLossDrawValueHead.
    """
    def __init__(self, spatial_size: int, c_in: int, c_hidden: int, n_hidden: int):
        super(WinLossValueHead, self).__init__()

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 2)

    def default_loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


class WinShareActionValueHead(Head):
    """
    WinShareActionValueHead is appropriate for games where the action-value head outputs win-share
    values. This is the case any times the game's ValueHead returns a tensor that gets transformed
    into a win-share array. This is the case for WinShareValueHead, WinLossDrawValueHead, and
    WinLossValueHead
    """
    def __init__(self, spatial_size: int, c_in: int, c_hidden: int, output_shape: Shape):
        super(WinShareActionValueHead, self).__init__()

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, math.prod(output_shape))

    def default_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.output_shape)
        return out


class WinShareValueHead(Head):
    """
    A head that produces a length-p logit vector, where p is the number of players in the game,
    usable for zero-sum games with p players for any p>=2. The k'th entry predicts the expected
    number of win-shares for the k'th player.

    Ideally, our value head predicts all possible outcomes, including all the various k-way draws
    for first-place, but there are some practical drawbacks to this:

    1. There 2^p - 1 possible outcomes, which can be a lot of logits to predict for large p.

    2. Some of those outcomes may be vanishingly rare, and having target categories that are too
       rare can lead to prediction trouble.

    Due to these reasons, WinShareValueHead is typically the best value head for multiplayer games.

    We could use CrossEntropyLoss, but it's arguably not appropriate given how we represent draws.
    We represent a draw between players 0 and 1 as a win-share target of [0.5, 0.5, 0, 0]. Even if
    the network perfectly predicts a logit vector that softmaxes to [0.5, 0.5, 0, 0], this yields
    nonzero loss.

    Instead, we use Kullback-Leibler Divergence (KL-Divergence) Loss.

    As for the head architecture, we follow WinLossDrawValueHead, simply changing the final
    output shape.
    """

    def __init__(self, spatial_size: int, c_in: int, c_hidden: int, n_hidden: int, shape: Shape):
        super(WinShareValueHead, self).__init__()

        (n_players, ) = shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_players)

    def default_loss_function(self):
        # TODO: change this to use KL-Divergence loss, as advertised above.
        #
        # (dshin) I briefly experimented with nn.KLDivLoss, but it didn't seem to work well. I think
        # I must have been using it incorrectly.
        return nn.CrossEntropyLoss()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


class ScoreHead(Head):
    """
    This maps to one of the subheads of ValueHead in the KataGo codebase.

    It outputs a (2, N) tensor, where the first row is a PDF (probability distribution function) and
    the second row is a CDF (cumulative distribution function).

    N represents the number of discrete score buckets.

    The loss is computed as the sum of a cross-entropy loss on the PDF and a mean-squared error loss
    on the CDF.

    See Section 4.1 of the KataGo paper for details.
    """
    def __init__(self, c_in: int, c_hidden: int, n_hidden: int, shape: Shape):
        super(ScoreHead, self).__init__()

        self.shape = shape
        assert shape[0] == 2, f'Unexpected shape {shape}'  # first dim is PDF/CDF
        assert len(shape) in (2, 3), f'Unexpected shape {shape}'

        self.shape = shape

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)

        self.gpool = GlobalPoolingLayer()

        n_gpool = GlobalPoolingLayer.NUM_CHANNELS * c_hidden
        self.linear1 = nn.Linear(n_gpool, n_hidden, bias=True)
        self.linear2 = nn.Linear(n_hidden, math.prod(shape), bias=True)

    def default_loss_function(self):
        pdf_loss_fn = nn.CrossEntropyLoss()
        cdf_loss_fn = nn.MSELoss()

        def loss(output: torch.Tensor, target: torch.Tensor):
            """
            Tensors are of shape (B, D, C, aux...), where:

            B = batch-size
            D = num distribution types (2: PDF, CDF)
            C = number of classes (i.e., number of possible scores)
            aux... = any number of additional dimensions

            aux... might be nonempty if for example we're predicting for multiple players
            """
            assert output.shape == target.shape, (output.shape, target.shape)
            assert output.shape[1] == 2, output.shape  # PDF, CDF

            pdf_loss = pdf_loss_fn(output[:, 0], target[:, 0])
            cdf_loss = cdf_loss_fn(output[:, 1], target[:, 1])

            return pdf_loss + cdf_loss

        return loss

    def forward(self, x):
        N = x.shape[0]

        out = x  # (N, C_in, H, W)
        out = self.conv(out)  # (N, C_hidden, H, W)
        out = self.act(out)  # (N, C_hidden, H, W)
        out = self.gpool(out)  # (N, n_gpool, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (N, n_gpool)

        out = self.linear1(out)  # (N, n_hidden)
        out = self.act(out)  # (N, n_hidden)
        out = self.linear2(out)  # (N, *)

        return out.view(N, *self.shape)  # (N, *shape)


class OwnershipHead(Head):
    """
    This maps to one of the subheads of ValueHead in the KataGo codebase.

    For historical reasons, I am mimicking the flow of the PolicyHead for now.
    """

    def __init__(self, c_in: int, c_hidden: int, shape: Shape):
        super(OwnershipHead, self).__init__()

        self.shape = shape
        assert len(shape) == 3, f'Unexpected shape {shape}, Conv2d will not work'
        n_categories = shape[0]

        self.act = F.relu
        self.conv1 = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(c_hidden, n_categories, kernel_size=1, bias=True)

    def default_loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out


class GeneralLogitHead(Head):
    """
    A head that produces an arbitrarily shaped tensor, unrelated to the input spatial dimensions.

    The outputs are interpeted as independent logits, which should be passed through a sigmoid to
    convert to independent probabilities.
    """
    def __init__(self, c_in: int, c_hidden: int, n_hidden: int, shape: Shape):
        super(GeneralLogitHead, self).__init__()

        self.shape = shape

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)

        self.gpool = GlobalPoolingLayer()

        n_gpool = GlobalPoolingLayer.NUM_CHANNELS * c_hidden
        self.linear1 = nn.Linear(n_gpool, n_hidden, bias=True)
        self.linear2 = nn.Linear(n_hidden, math.prod(shape), bias=True)

    def default_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def forward(self, x):
        N = x.shape[0]

        out = x  # (N, C_in, H, W)
        out = self.conv(out)  # (N, C_hidden, H, W)
        out = self.act(out)  # (N, C_hidden, H, W)
        out = self.gpool(out)  # (N, n_gpool, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (N, n_gpool)

        out = self.linear1(out)  # (N, n_hidden)
        out = self.act(out)  # (N, n_hidden)
        out = self.linear2(out)  # (N, *)

        return out.view(N, *self.shape)  # (N, *shape)


MODULE_MAP = {
    'ConvBlock': ConvBlock,
    'ConvBlockWithGlobalPooling': ConvBlockWithGlobalPooling,
    'GeneralLogitHead': GeneralLogitHead,
    'KataGoNeck': KataGoNeck,
    'OwnershipHead': OwnershipHead,
    'ResBlock': ResBlock,
    'ResBlockWithGlobalPooling': ResBlockWithGlobalPooling,
    'PolicyHead': PolicyHead,
    'ScoreHead': ScoreHead,
    'TransformerBlock': TransformerBlock,
    'WinLossDrawValueHead': WinLossDrawValueHead,
    'WinLossDrawValueUncertaintyHead': WinLossDrawValueUncertaintyHead,
    'WinLossValueHead': WinLossValueHead,
    'WinShareValueHead': WinShareValueHead,
    'WinShareActionValueHead': WinShareActionValueHead,
    }
