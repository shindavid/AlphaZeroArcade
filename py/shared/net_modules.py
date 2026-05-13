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
from shared.backup_net import BackupNet
from shared.transformer_modules import TransformerBlock
from util.torch_util import LossFunction, Shape

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

import abc
import logging
import math
from typing import Optional, Type


logger = logging.getLogger(__name__)


class Head(nn.Module):
    def requires_policy_scaling(self) -> bool:
        """
        If True, then the loss term for this head should be scaled by A / B, where:

        - A is the number of legal actions for the position
        - B is the total number of actions in the action space
        """
        return False

    def default_loss_function(self) -> Optional[Type[LossFunction]]:
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
        super().__init__()

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

    def __init__(self, input_shape: Shape, output_shape: Shape):
        super().__init__()

        c_in = input_shape[0]
        c_out = output_shape[0]

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
        super().__init__()
        self.norm = nn.BatchNorm2d(c_in)
        self.act = F.relu

    def forward(self, x):
        out = x
        out = self.act(self.norm(out))
        return out


class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """
    Copied from KataGo's SoftPlusWithGradientFloorFunction in py/model_pytorch.py.

    Same as softplus, except on backward pass, we never let the gradient decrease below grad_floor.
    Equivalent to having a dynamic learning rate depending on stop_grad(x) where x is the input.
    If square, then also squares the result while halving the input, and still also keeping the same gradient.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_floor: float, square: bool):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor # grad_floor is not a tensor
        if square:
            return torch.square(torch.nn.functional.softplus(0.5 * x))
        else:
            return torch.nn.functional.softplus(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_floor = ctx.grad_floor
        grad_x = None
        grad_grad_floor = None
        grad_square = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, grad_grad_floor, grad_square

    @staticmethod
    def symbolic(g, x, grad_floor, square):
        # ONNX export override.
        #
        # We deliberately AVOID emitting the ONNX `Softplus` op here. TensorRT 10.11.0's Myelin
        # backend has a bug where it segfaults during engine build when fusing the `Softplus`
        # op as it appears in the value_uncertainty / action_value_uncertainty head subgraphs of
        # our beta0 inference network. By decomposing softplus into the numerically stable
        # primitive form
        #
        #     softplus(t) = relu(t) + log1p(exp(-|t|))
        #                 = max(t, 0) + log(1 + exp(-|t|))
        #
        # we sidestep the offending fusion entirely. The forward / backward math at training
        # time is unchanged; only the exported graph differs.
        #
        # If TensorRT fixes this upstream, this method can be removed (and we'd also drop the
        # corresponding `setBuilderOptimizationLevel(1)` workaround in cpp/src/core/NeuralNet.cpp).
        # `grad_floor` is unused at inference time. `square` is a Python bool.
        del grad_floor

        # Match input dtype/shape for the constants by using FLOATs scalars; ONNX broadcasts.
        one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))

        if square:
            half = g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.float32))
            t = g.op("Mul", x, half)
        else:
            t = x

        relu_t = g.op("Relu", t)
        abs_t = g.op("Abs", t)
        neg_abs = g.op("Neg", abs_t)
        exp_neg_abs = g.op("Exp", neg_abs)
        log1p = g.op("Log", g.op("Add", exp_neg_abs, one))
        sp = g.op("Add", relu_t, log1p)

        if square:
            return g.op("Mul", sp, sp)
        return sp


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
    def __init__(self, in_shape: Shape, c_gpool: int, out_shape: Shape):
        super().__init__()

        c_in = in_shape[0]
        c_out = out_shape[0]

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
    def __init__(self, shape: Shape, mid_shape: Shape):
        super().__init__()

        self.conv1 = ConvBlock(shape, mid_shape)
        self.conv2 = ConvBlock(mid_shape, shape)

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
    def __init__(self, shape: Shape, c_mid_gp: int, mid_shape: Shape):
        super().__init__()

        c = shape[0]
        c_mid = mid_shape[0]

        assert 0 < c_mid_gp < c_mid

        mid_shape_adjusted = (c_mid - c_mid_gp, *mid_shape[1:])

        self.conv1 = ConvBlockWithGlobalPooling(shape, c_mid_gp, mid_shape_adjusted)
        self.conv2 = ConvBlock(mid_shape_adjusted, shape)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return x + out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, board_size, dropout=0.):
        super().__init__()
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
    def __init__(self, trunk_shape: Shape, c_hidden: int, output_shape: Shape):
        super().__init__()

        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, math.prod(output_shape))

    def default_loss_function(self):
        return nn.CrossEntropyLoss

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.output_shape)
        return out


class ValueHeadBase(Head, abc.ABC):
    """
    Base class for value heads.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def to_win_share(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert the raw output of the value head to win-share values.

        The input y is the raw output of the value head, of shape (B, value_dim), where value_dim
        is specific to the particular value head implementation.

        The output is of shape (B, n_players), where n_players is the number of players in the game.
        The k'th entry in the last dimension is the predicted win-share for the k'th player. The
        entries should be nonnegative and sum to 1.
        """
        pass


class WinLossDrawValueHead(ValueHeadBase):
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
    def __init__(self, trunk_shape: Shape, c_hidden: int, n_hidden: int):
        super().__init__()

        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 3)

    def default_loss_function(self):
        return nn.CrossEntropyLoss

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out

    def to_win_share(self, y: torch.Tensor) -> torch.Tensor:
        wld = y.softmax(dim=-1)  # (B, 3)
        return wld[:, :2] + 0.5 * wld[:, 2:]


class ValueUncertaintyHead(Head):
    """
    A head that produces a length-1 vector, usable for 2-player games that permit draws.

    This is an uncertainty head that can be used in conjunction with WinLossDrawValueHead to
    represent uncertainty in the value prediction. The output is interpreted as delta^2, where delta
    is the difference between the value prediction (converted from W/L/D to win-shares) and the true
    value.

    See the "Uncertainty-Weighted MCTS Playouts" section here:

    https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md

    NOTE: although for zero-sum 2-player games, a single scalar suffices to represent uncertainty, a
    single scalar is not appropriate for multiplayer games. For example, in a 4-player game, player
    1 might be very certain that they will lose, while players 2, 3, and 4 might be very uncertain
    about their outcomes. In general, in a zero-sum p-player game, the space of possible
    uncertainties is (p-1)-dimensional. I haven't come up with a good way to represent this yet.
    """
    def __init__(
        self,
        trunk_shape: Shape,
        c_hidden: int,
        n_hidden: int,
        output_shape: Shape
    ):
        super().__init__()

        c_trunk, H, W = trunk_shape
        self.out_dim = output_shape[0]
        self.act = F.relu

        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, self.out_dim)

    def default_loss_function(self):
        # KataGo uses delta=0.4 for value targets.
        #
        # Claude recommends 0.1 to adjust for the fact that our values are in the range [0, 1]
        # rather than [-1, +1]. Detailed explanation:
        #
        # KataGo's delta=0.4 was chosen for value targets in [0, 4] (so delta ≈ 10% of the max
        # target). For your [0, 1] value targets, delta = 0.1 matches that ratio.
        return lambda : nn.HuberLoss(delta=0.1)

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        t = self.conv(trunk)           # (B, c_hidden, H, W)
        t = self.act(t)
        t = t.mean(dim=(2, 3))         # global average pool → (B, c_hidden)
        t = self.act(self.linear1(t))  # (B, n_hidden)
        h = self.linear2(t)            # (B, out_dim)

        # The .05 multiplier is here is based on KataGo.
        #
        # Claude says: KataGo uses 0.25 for value, where targets live in [0, 4]. Your value targets
        # live in [0, 1] (since win-shares are in [0, 1]), so a 4× smaller multiplier ≈ 0.0625 keeps
        # the same ratio. Round to 0.05 or 0.1 as a starting point.
        #
        # I asked what the point of this multiplier is - can't it just be absorbed into linear2?
        # Claude answers:
        #
        # In principle yes, in practice no — and this is a subtle but important point. The
        # multiplier sets the scale at initialization. With a freshly initialized linear2
        # (typical Kaiming/Xavier weights → outputs near 0, softplus(0) ≈ 0.69, squared softplus ≈
        # 0.48), without the multiplier you'd start by predicting uncertainty ≈ 0.48 everywhere.
        # Your actual targets are sq-errors that are typically ~0.001 – 0.01 early in training. The
        # head would need to grow linear2 weights very negative to drag the softplus output down —
        # and during that adjustment phase, the gradient through softplus is shrinking (it saturates
        # toward 0 on the negative side), so learning slows down exactly when you want it to keep
        # moving. The gradient-floor mechanism partially fixes that, but it's much cleaner to just
        # start in roughly the right output range.
        #
        # It's the same argument as initializing the bias of a regression head to the mean of the
        # target. The multiplier is a (deliberate, fixed) "init-scale prior."
        #
        # For your case: typical squared error of value predictions in C4 should also be small
        # (~0.01 order of magnitude in steady state, larger early on). 0.05 is reasonable — maybe a
        # touch large. KataGo's 0.25 for value gives a typical "default" output of ~0.25 × 0.48 ≈
        # 0.12 at init, against targets that average ~`0.05. Your 0.05 × 0.48 ≈ 0.024 against
        # targets ~0.05` is in the same ballpark. Fine.
        return SoftPlusWithGradientFloorFunction.apply(h, .05, True) * .05


class WinLossValueHead(ValueHeadBase):
    """
    A head that produces a length-2 logit vector, usable for 2-player games that do NOT permit
    draws.

    This is based off WinLossDrawValueHead.
    """
    def __init__(self, trunk_shape: Shape, c_hidden: int, n_hidden: int):
        super().__init__()

        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 2)

    def default_loss_function(self):
        return nn.CrossEntropyLoss

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out

    def to_win_share(self, y: torch.Tensor) -> torch.Tensor:
        return y.softmax(dim=-1)


class WinShareActionValueHead(Head):
    """
    WinShareActionValueHead is appropriate for games where the action-value head outputs win-share
    values. This is the case any times the game's ValueHead returns a tensor that gets transformed
    into a win-share array. This is the case for WinShareValueHead, WinLossDrawValueHead, and
    WinLossValueHead
    """
    def __init__(self, trunk_shape: Shape, c_hidden: int, output_shape: Shape):
        super().__init__()

        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, math.prod(output_shape))

    def requires_policy_scaling(self) -> bool:
        return True

    def default_loss_function(self):
        return nn.KLDivLoss

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.output_shape)
        return F.log_softmax(out, dim=-1)  # KLDivLoss expects log-probabilities


class WinShareValueHead(ValueHeadBase):
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

    def __init__(self, trunk_shape: Shape, c_hidden: int, n_hidden: int, shape: Shape):
        super().__init__()

        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        (n_players, ) = shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_players)

    def default_loss_function(self):
        # TODO: change this to use KL-Divergence loss, as advertised above.
        #
        # (dshin) I briefly experimented with nn.KLDivLoss, but it didn't seem to work well. I think
        # I must have been using it incorrectly.
        return nn.CrossEntropyLoss

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out

    def to_win_share(self, y: torch.Tensor) -> torch.Tensor:
        return y.softmax(dim=-1)


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
    def __init__(self, trunk_shape: Shape, c_hidden: int, n_hidden: int, shape: Shape):
        super().__init__()

        c_trunk = trunk_shape[0]

        self.shape = shape
        assert shape[0] == 2, f'Unexpected shape {shape}'  # first dim is PDF/CDF
        assert len(shape) in (2, 3), f'Unexpected shape {shape}'

        self.shape = shape

        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)

        self.gpool = GlobalPoolingLayer()

        n_gpool = GlobalPoolingLayer.NUM_CHANNELS * c_hidden
        self.linear1 = nn.Linear(n_gpool, n_hidden, bias=True)
        self.linear2 = nn.Linear(n_hidden, math.prod(shape), bias=True)

    def default_loss_function(self):
        pdf_loss_fn = nn.CrossEntropyLoss()
        cdf_loss_fn = nn.MSELoss()

        class Loss:
            def __call__(self, output: torch.Tensor, target: torch.Tensor):
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

        return Loss

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

    def __init__(self, trunk_shape: Shape, c_hidden: int, shape: Shape):
        super().__init__()

        c_trunk = trunk_shape[0]

        self.shape = shape
        assert len(shape) == 3, f'Unexpected shape {shape}, Conv2d will not work'
        n_categories = shape[0]

        self.act = F.relu
        self.conv1 = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(c_hidden, n_categories, kernel_size=1, bias=True)

    def default_loss_function(self):
        return nn.CrossEntropyLoss

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out


class ActionValueUncertaintyHead(Head):
    """
    A head that produces action value uncertainty values. It aims to predict what the
    ValueUncertaintyHead would predict for the game state that would result after taking each
    action.
    """
    def __init__(self, trunk_shape: Shape, c_hidden: int, output_shape: Shape):
        super().__init__()
        c_trunk, H, W = trunk_shape

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * H * W, math.prod(output_shape))

    def requires_policy_scaling(self) -> bool:
        return True

    def default_loss_function(self):
        # See comments in ValueUncertainHead.default_loss_function()
        return lambda **kwargs: nn.HuberLoss(delta=0.1, **kwargs)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.output_shape)

        # see comments in ValueUncertainHead.forward()
        return SoftPlusWithGradientFloorFunction.apply(out, .05, True) * .05


class StaticLatentHead(Head):
    """
    The static per-node latent head z_s used by BetaZero's BackupNet.

    Architecture: trunk -> 1x1 conv (c_hidden channels) -> ReLU -> spatial mean pool ->
                  Linear(c_hidden, latent_dim).

    z_s feeds into the BackupNet (CPU-side NNUE) at the layer-1 stage. It has no direct loss
    term; gradient flows back from the BackupNet into this head's own conv/linear weights, but
    is detached at the trunk boundary so BackupNet loss does not pollute trunk learning. At
    inference time, z_s is computed once at node expansion on the GPU and stored at the node,
    since it is static per node.

    See docs/BetaZero.pdf, Sections 4.2 and 4.3.
    """

    def __init__(self, trunk_shape: Shape, c_hidden: int, latent_dim: int):
        super().__init__()
        c_trunk = trunk_shape[0]
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden, latent_dim)

    def default_loss_function(self):
        return None

    def forward(self, x):
        x = x.detach()          # stop BackupNet gradient from polluting the trunk
        h = self.conv(x)
        h = self.act(h)
        h = h.mean(dim=(2, 3))  # global average pool -> (B, c_hidden)
        return self.linear(h)   # (B, latent_dim)


class ActionLatentHead(Head):
    """
    The per-action latent head action_latent used by BetaZero's BackupNet.

    Architecture mirrors PolicyHead: trunk -> 1x1 conv (c_hidden) -> ReLU -> flatten ->
                                     Linear -> reshape to output_shape.

    output_shape is (A, action_latent_dim), where A is the number of actions in the action space and
    action_latent_dim is the per-action latent dimension. Internal specialization for games where the
    action space matches the spatial shape (e.g. policy-as-conv) may be added later as an
    internal optimization.

    action_latent feeds into ChildEmbeddingHead alongside the per-action [Qs, Ws, N, P, AVs, AUs] tuple.
    It has no direct loss term; gradient flows back from the BackupNet into this head's own
    conv/linear weights, but is detached at the trunk boundary so BackupNet loss does not
    pollute trunk learning. (ChildEmbeddingHead's only trunk-connected parent is action_latent, so this
    detach is sufficient to sever the BackupNet -> trunk path through the per-action route.)

    See docs/BetaZero.pdf, Sections 4.2 and 4.3.
    """

    def __init__(self, trunk_shape: Shape, c_hidden: int, output_shape: Shape):
        super().__init__()
        c_trunk = trunk_shape[0]
        spatial_size = math.prod(trunk_shape[1:])

        self.output_shape = output_shape
        self.act = F.relu
        self.conv = nn.Conv2d(c_trunk, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, math.prod(output_shape))

    def default_loss_function(self):
        return None

    def forward(self, x):
        x = x.detach()          # stop BackupNet gradient from polluting the trunk
        out = self.conv(x)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out.view(-1, *self.output_shape)


class ChildEmbeddingHead(Head):
    """
    The per-child embedding head used by BetaZero's NNUE backup.

    For each action i, computes:

        x_i = [child_stats_i ; action_latent,i]                       # length child_stat_dim + action_latent_dim
        e_i = ReLU(W_e @ x_i + b_e) * (P_i > 0)             # length embed_dim

    where `child_stats_i` is the per-action tuple [Qs_i, Ws_i, N_i, P_i, AVs_i, AUs_i] and action_latent
    comes from ActionLatentHead. The C++ side assembles `child_stats` (with N_i = 0 for unvisited
    children at expansion, post-Dirichlet-noise P_i, the action-value-head estimates AVs/AUs at
    parent-eval time, etc.) and feeds it as the `input_child_stats` Model input.

    Output shape: (B, A, embed_dim).
    Per-action invalid actions (P_i = 0) are masked out via multiplication by (P > 0).

    The W_e, b_e weights are also exported as orphan ONNX initializers (`nnue/child_embed.*`)
    so the C++ NNUE engine can do subtract-add updates without parsing ONNX op weights.

    See docs/BetaZero.pdf, Sections 4.3 and 7.1.
    """

    # Per-child stat layout: [Qs_i, Ws_i, N_i, P_i, AVs_i, AUs_i].
    CHILD_STAT_DIM = 6
    P_INDEX = 3

    def __init__(
        self,
        child_stats_shape: Shape,            # (A, 6)  [Qs, Ws, N, P, AVs, AUs]
        action_latent_shape: Shape,          # (A, action_latent_dim)
        embed_dim: int,
    ):
        super().__init__()
        assert len(child_stats_shape) == 2 and child_stats_shape[1] == self.CHILD_STAT_DIM, (
            f'child_stats_shape must be (A, {self.CHILD_STAT_DIM}), got {child_stats_shape}')
        A = child_stats_shape[0]
        assert action_latent_shape[0] == A, (action_latent_shape, A)

        action_latent_dim = math.prod(action_latent_shape[1:]) if len(action_latent_shape) > 1 else 1
        per_child_in = self.CHILD_STAT_DIM + action_latent_dim

        self.A = A
        self.action_latent_dim = action_latent_dim
        self.embed_dim = embed_dim
        self.child_embed = nn.Linear(per_child_in, embed_dim)

    def default_loss_function(self):
        return None

    def forward(self, child_stats, action_latent):
        # child_stats: (B, A, CHILD_STAT_DIM)
        # action_latent: (B, A, action_latent_dim) or (B, A) if action_latent_dim == 1
        B = child_stats.shape[0]
        A = self.A
        if action_latent.dim() == 2:
            assert self.action_latent_dim == 1
            action_latent = action_latent.unsqueeze(-1)
        za = action_latent.reshape(B, A, self.action_latent_dim)

        x = torch.cat([child_stats, za], dim=-1)               # (B, A, per_child_in)
        e = F.relu(self.child_embed(x))                         # (B, A, embed_dim)
        p = child_stats[..., self.P_INDEX:self.P_INDEX + 1]    # (B, A, 1)
        mask = (p > 0).to(e.dtype)
        return e * mask                                         # (B, A, embed_dim)

    def collect_graph_initializers(self, out):
        for name, param in self.named_parameters():
            out[name] = param.detach().cpu().numpy().astype(np.float32, copy=False)


class AccumulatorHead(Head):
    """
    The per-node accumulator head: sum over actions of the per-child embeddings produced by
    ChildEmbeddingHead.

    No trainable parameters. Forward simply sums the masked-by-(P>0) per-action embeddings
    over the action dimension. Output shape: (B, embed_dim).

    The C++ side stores this initial accumulator on each Node at expansion. Subsequent backups
    use NNUE-style subtract-add updates on this accumulator, driven by per-Edge `e_i` storage
    and the `nnue/child_embed.*` orphan initializers.

    See docs/BetaZero.pdf, Section 7.1.
    """

    def __init__(self):
        super().__init__()

    def default_loss_function(self):
        return None

    def forward(self, e):
        # e: (B, A, embed_dim) from ChildEmbeddingHead, already masked by (P > 0).
        return e.sum(dim=1)


MODULE_MAP = {
    'AccumulatorHead': AccumulatorHead,
    'ActionLatentHead': ActionLatentHead,
    'ActionValueUncertaintyHead': ActionValueUncertaintyHead,
    'BackupNet': BackupNet,
    'ChildEmbeddingHead': ChildEmbeddingHead,
    'ConvBlock': ConvBlock,
    'ConvBlockWithGlobalPooling': ConvBlockWithGlobalPooling,
    'KataGoNeck': KataGoNeck,
    'OwnershipHead': OwnershipHead,
    'ResBlock': ResBlock,
    'ResBlockWithGlobalPooling': ResBlockWithGlobalPooling,
    'PolicyHead': PolicyHead,
    'ScoreHead': ScoreHead,
    'StaticLatentHead': StaticLatentHead,
    'TransformerBlock': TransformerBlock,
    'ValueUncertaintyHead': ValueUncertaintyHead,
    'WinLossDrawValueHead': WinLossDrawValueHead,
    'WinLossValueHead': WinLossValueHead,
    'WinShareValueHead': WinShareValueHead,
    'WinShareActionValueHead': WinShareActionValueHead,
    }
