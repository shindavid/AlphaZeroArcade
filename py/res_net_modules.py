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
import copy
from dataclasses import dataclass, field
import math
import os
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn as nn
from torch.nn import functional as F

from neural_net import LearningTarget, OwnershipTarget, PolicyTarget, ScoreMarginTarget, ValueTarget
from util.torch_util import Shape


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
    def __init__(self, name: str, c_in_out: int, c_mid: int):
        super(ResBlock, self).__init__()
        self.name = name
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


class Neck(nn.Module):
    """
    A layer connecting the output of the residual blocks to the heads (policy, value, and aux). This
    does not exist in KataGo; we construct it by pulling out components from KataGo's ValueHead
    and PolicyHead. Specifically, we have this layer compute 3 components:

    S:  spatial info
    G:  globally pooled info
    SG: spacial info mixed with globally pooled info

    We introduce this layer for increased modularity with respect to the heads.
    """
    def __init__(self, c_in: int, c_spatial: int, c_gpool: int):
        super(Neck, self).__init__()

        self.act = F.relu
        self.conv_s = nn.Conv2d(c_in, c_spatial, kernel_size=1, bias=False)
        self.conv_g = nn.Conv2d(c_in, c_gpool, kernel_size=1, bias=True)
        self.pool_g = GlobalPoolingLayer()
        self.linear_g = nn.Linear(GlobalPoolingLayer.NUM_CHANNELS * c_gpool, c_spatial, bias=True)

    def forward(self, x):
        out_s = self.conv_s(x)

        out_g = self.conv_g(x)
        out_g = self.act(out_g)
        out_g = self.pool_g(out_g).squeeze(-1).squeeze(-1)

        out_sg = out_s + self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)
        out_sg = self.act(out_sg)
        return out_s, out_g, out_sg


class Head(nn.Module):
    def __init__(self, name: str, target: LearningTarget):
        super(Head, self).__init__()

        self.name = name
        self.target = target


class PolicyHead(Head):
    """
    This, together with Neck, corresponds to the policy-subhead of PolicyHead in the KataGo
    codebase.

    KataGo uses a Conv2d for its final layer, with special handling for the pass move. We currently
    match AlphaGo and use a Linear layer instead, as this generalizes better to games where the
    policy shape does not match the board shape. Later we should make this more flexible so that
    we can match KataGo if we desire.

    For reference, the AlphaGo Zero paper describes the policy head as follows:

    1. A convolution of 2 filters of kernel size 1 x 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19^2 + 1 = 362 corresponding to
    logit probabilities for all intersections and the pass move
    """
    def __init__(self, name: str, board_size: int, c_in: int, policy_shape: Union[Shape, int]):
        super(PolicyHead, self).__init__(name, PolicyTarget())

        policy_shape = tuple([policy_shape]) if isinstance(policy_shape, int) else policy_shape
        self.policy_shape = policy_shape
        self.policy_size = math.prod(policy_shape)
        self.board_size = board_size
        self.c_in = c_in
        self.linear = nn.Linear(c_in * board_size, self.policy_size)
        self.conv = nn.Conv2d(c_in, self.policy_size, kernel_size=1, bias=False)

    def forward(self, s, g, sg):
        out = sg
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.policy_shape)
        return out


class ValueHead(Head):
    """
    This, together with Neck, corresponds to the value-subhead ValueHead in the KataGo codebase.

    Note that while AlphaGo used a scalar output for the value head, KataGo uses a length-3 logit
    vector, corresponding to {win, loss, draw} probabilities.

    Because we support p-player games, generalizing KataGo's approach seems like it might be
    unwieldy, as we would need exponentially many outputs corresponding to all the different
    possible draw-combinations that could occur in arbitrary p-player games. We instead generalize
    AlphaGo's representation, using a length-p logit vector, corresponding to each player's
    expected win-share.

    For reference, the AlphaGo Zero paper describes the value head as follows:

    1. A convolution of 1 filter of kernel size 1 x 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    def __init__(self, name: str, c_in: int, n_players: int):
        super(ValueHead, self).__init__(name, ValueTarget())

        self.linear = nn.Linear(c_in, n_players)

    def forward(self, s, g, sg):
        return self.linear(g)


class ScoreMarginHead(Head):
    def __init__(self, name: str, c_in: int, c_hidden: int, max_score_margin: int,
                 min_score_margin: Optional[int]=None):
        target = ScoreMarginTarget(max_score_margin, min_score_margin)
        super(ScoreMarginHead, self).__init__(name, target)

        min_score_margin = -max_score_margin if min_score_margin is None else min_score_margin
        n_possible_score_margins = max_score_margin - min_score_margin + 1
        self.linear1 = nn.Linear(c_in, c_hidden)
        self.linear2 = nn.Linear(c_hidden, n_possible_score_margins)
        self.act = F.relu

    def forward(self, s, g, sg):
        out = g
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


class OwnershipHead(Head):
    def __init__(self, name: str, c_in: int, n_possible_owners: int):
        super(OwnershipHead, self).__init__(name, OwnershipTarget())

        self.conv = nn.Conv2d(c_in, n_possible_owners, kernel_size=1, bias=False)

    def forward(self, s, g, sg):
        """
        We have this operate on s, rather than the richer sg, to match KataGo.

        TODO: try operating on sg instead. If this is not worse, then we can remove s from the
        Neck output.
        """
        return self.conv(s)


MODULE_MAP = {
    'ConvBlock': ConvBlock,
    'ConvBlockWithGlobalPooling': ConvBlockWithGlobalPooling,
    'ResBlock': ResBlock,
    'ResBlockWithGlobalPooling': ResBlockWithGlobalPooling,
    'Neck': Neck,
    'PolicyHead': PolicyHead,
    'ValueHead': ValueHead,
    'ScoreMarginHead': ScoreMarginHead,
    'OwnershipHead': OwnershipHead,
    }


@dataclass
class ModuleSpec:
    type: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    input_shape: Shape
    stem: ModuleSpec
    blocks: List[ModuleSpec]
    neck: ModuleSpec
    heads: List[ModuleSpec]
    loss_weights: Dict[str, float]

    def validate(self):
        for spec in [self.stem] + self.blocks + [self.neck] + self.heads:
            assert spec.type in MODULE_MAP, f'Unknown module type {spec.type}'


ModelConfigGenerator = Callable[[Shape], ModelConfig]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()

        config.validate()

        self.config = config
        self.input_shape = config.input_shape
        self.stem = Model._construct_module(config.stem)
        self.blocks = nn.ModuleList(map(Model._construct_module, config.blocks))
        self.neck = Model._construct_module(config.neck)
        self.heads = nn.ModuleList(map(Model._construct_module, config.heads))
        self.loss_weights = config.loss_weights

        self.validate()

    @property
    def learning_targets(self) -> List[LearningTarget]:
        return [head.target for head in self.heads]

    @property
    def target_names(self) -> List[str]:
        return [head.name for head in self.heads]

    def forward(self, x):
        out = x
        out = self.stem(out)
        for block in self.blocks:
            out = block(out)
        out = self.neck(out)
        return tuple(head(*out) for head in self.heads)

    @staticmethod
    def _construct_module(spec: ModuleSpec) -> nn.Module:
        cls = MODULE_MAP[spec.type]
        return cls(*spec.args, **spec.kwargs)

    def validate(self):
        head_names = set()
        for head in self.heads:
            assert head.name not in head_names, f'Head with name {head.name} already exists'
            head_names.add(head.name)

        assert self.heads[0].name == 'policy', 'The first head must be the policy head'
        assert self.heads[1].name == 'value', 'The second head must be the value head'

        for name in self.loss_weights:
            assert name in head_names, f'Loss weight for unknown head {name}'

        for name in head_names:
            assert name in self.loss_weights, f'Loss weight missing for head {name}'

    def validate_targets(self, targets: List[str]):
        for target in targets:
            assert target in self.loss_weights, f'Unknown target {target}'
        for target in self.loss_weights:
            assert target in targets, f'Missing target {target}'

    @classmethod
    def load_model(cls, filename: str, device = 'cuda', verbose: bool = False,
                   eval_mode: bool = True,
                   set_grad_enabled: Optional[bool] = None) -> torch.jit.ScriptModule:
        """
        Loads a model previously saved to disk via save(). This uses torch.jit.load(), which
        returns a torch.jit.ScriptModule, which looks/feels/sounds like nn.Module, but is not
        exactly the same thing.

        If set_grad_enabled is None (default), then calls torch.set_grad_enabled(False) if
        eval_mode is True. Note that this mutates torch's global state.
        """
        if verbose:
            print(f'Loading model from {filename}')

        net = torch.jit.load(filename)
        net.to(device)
        set_grad_enabled = not eval_mode if set_grad_enabled is None else set_grad_enabled
        if set_grad_enabled:
            torch.set_grad_enabled(False)

        if eval_mode:
            net.eval()
        else:
            net.train()

        if verbose:
            print(f'Model successfully loaded!')
        return net

    def save_model(self, filename: str, verbose: bool = False):
        """
        Saves this network to disk, from which it can be loaded either by c++ or by python. Uses the
        torch.jit.trace() function to accomplish this.

        Note that prior to saving, we "freeze" the model, by switching it to eval mode and disabling
        gradient. The documentation seems to imply that this is an important step:

        "...In the returned :class:`ScriptModule`, operations that have different behaviors in
          ``training`` and ``eval`` modes will always behave as if it is in the mode it was in
          during tracing, no matter which mode the `ScriptModule` is in..."

        In order to avoid modifying self during the save() call, we actually deepcopy self and then
        do the freeze and trace on the copy.
        """
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        clone = copy.deepcopy(self)

        # strip all aux heads to avoid unnecessary c++ computation
        clone.heads = clone.heads[:2]

        clone.to('cpu')
        clone.eval()
        forward_shape = tuple([1] + list(self.input_shape))
        example_input = torch.zeros(forward_shape)
        mod = torch.jit.trace(clone, example_input)
        mod.save(filename)
        if verbose:
            print(f'Model saved to {filename}')

    @staticmethod
    def load_from_checkpoint(checkpoint: Dict[str, Any]) -> 'Model':
        """
        Load a model from a checkpoint. Inverse of add_to_checkpoint().
        """
        model_state_dict = checkpoint['model.state_dict']
        config = checkpoint['model.config']
        model = Model(config)
        model.load_state_dict(model_state_dict)
        return model

    def add_to_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Save the current state of this neural net to a checkpoint, so that it can be loaded later
        via load_from_checkpoint().
        """
        checkpoint.update({
            'model.state_dict': self.state_dict(),
            'model.config': self.config,
        })
