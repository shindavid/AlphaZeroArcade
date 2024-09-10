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
import pickle
import tempfile
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn as nn
from torch.nn import functional as F

from shared.learning_targets import ActionValueTarget, LearningTarget, OwnershipTarget, \
    PolicyTarget, ScoreTarget, ValueTarget
from util.repo_util import Repo
from util.torch_util import Shape


@dataclass
class ShapeInfo:
    name: str
    target_index: int
    shape: Shape


ShapeInfoDict = Dict[str, ShapeInfo]


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


class Head(nn.Module):
    def __init__(self, name: str, target: LearningTarget):
        super(Head, self).__init__()

        self.name = name
        self.target = target


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
    def __init__(self, name: str, spatial_size: int, c_in: int, c_hidden: int, policy_size: int,
                 target_cls=PolicyTarget):
        super(PolicyHead, self).__init__(name, target_cls())

        policy_shape = (policy_size, )
        self.policy_shape = policy_shape
        self.policy_size = policy_size

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear = nn.Linear(c_hidden * spatial_size, self.policy_size)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = out.view(-1, *self.policy_shape)
        return out


class ActionValueHead(PolicyHead):
    def __init__(self, *args, **kwargs):
        super(ActionValueHead, self).__init__(*args, **kwargs, target_cls=ActionValueTarget)


class ValueHead(Head):
    """
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

    Note that while AlphaGo used a scalar output for the value head, KataGo uses a length-3 logit
    vector, corresponding to {win, loss, draw} probabilities.

    Because we support p-player games, generalizing KataGo's approach seems like it might be
    unwieldy, as the number of different win/loss/draw combinations in general p-player games is
    exponential in p. We instead go with a sort of hybrid approach, using a length-p logit vector,
    corresponding to each player's expected win-share.

    It is worth mentioning that some time ago, David Wu suggested relaxing various assumptions,
    like zero-sumness, and like fixed-sized rewards for winning/losing. If/when we explore this,
    we may want to reexamine our expected win-share representation.

    TODO: having 2 linear layers feels unnecessary. Try removing one, perhaps increasing
    c_hidden to compensate.
    """
    def __init__(self, name: str, spatial_size: int, c_in: int, c_hidden: int, n_hidden: int,
                 n_players: int):
        super(ValueHead, self).__init__(name, ValueTarget())

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.linear1 = nn.Linear(c_hidden * spatial_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_players)

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

    The design of this mostly matches what is described in the KataGo paper. The scaling component
    is something I am not sure about - whether it is only needed because KataGo handles multiple
    board sizes, or if it something that is generally useful. I have included it for now, but it is
    worth experimenting with removing it.
    """
    def __init__(self, name: str, c_in: int, c_hidden: int, n_hidden: int, shape: Shape):
        super(ScoreHead, self).__init__(name, ScoreTarget())

        self.shape = shape
        assert shape[0] == 2, f'Unexpected shape {shape}'  # first dim is PDF/CDF
        assert len(shape) in (2, 3), f'Unexpected shape {shape}'

        self.shape = shape
        self.n_scores = shape[1]
        self.last_dim = shape[0] * math.prod(shape[2:])

        self.act = F.relu
        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)

        self.gpool = GlobalPoolingLayer()

        n_gpool = GlobalPoolingLayer.NUM_CHANNELS * c_hidden
        self.scale_linear1 = nn.Linear(n_gpool, n_hidden, bias=True)
        self.scale_linear2 = nn.Linear(n_hidden, 1, bias=True)
        self.softplus = nn.Softplus()

        self.linear1 = nn.Linear(n_gpool + 1, n_hidden, bias=True)
        self.linear2 = nn.Linear(n_hidden, self.last_dim, bias=True)

        n_scores = self.n_scores
        self.constant_row = torch.arange(n_scores).reshape((1, n_scores, 1)) * (1.0 / n_scores)

    def forward(self, x):
        N = x.shape[0]
        n_scores = self.n_scores
        last_dim = self.last_dim

        out = x  # (N, C_in, H, W)
        out = self.conv(out)  # (N, C_hidden, H, W)
        out = self.act(out)  # (N, C_hidden, H, W)
        out = self.gpool(out)  # (N, n_gpool, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (N, n_gpool)

        scale = self.scale_linear1(out)  # (N, n_hidden)
        scale = self.act(scale)  # (N, n_hidden)
        scale = self.scale_linear2(scale)  # (N, 1)
        scale = self.softplus(scale)  # (N, 1)
        scale = scale.unsqueeze(2)  # (N, 1, 1)
        scale = scale.expand(-1, n_scores, last_dim)  # (N, n_scores, last_dim)

        out_s = out.unsqueeze(1)  # (N, 1, n_gpool)
        out_s = out_s.expand(-1, n_scores, -1)  # (N, n_scores, n_gpool)

        constant_plane = self.constant_row.to(x.device).expand(N, -1, -1)  # (N, n_scores, 1)
        out_s = torch.cat([out_s, constant_plane], dim=2)  # (N, n_scores, n_gpool + 1)
        out_s = self.linear1(out_s)  # (N, n_scores, n_hidden)
        out_s = self.act(out_s)  # (N, n_scores, n_hidden)
        out_s = self.linear2(out_s)  # (N, n_scores, last_dim)

        out = out_s * scale  # (N, n_scores, last_dim)
        out = out.view(N, *self.shape)  # (N, *shape)
        return out


class OwnershipHead(Head):
    """
    This maps to one of the subheads of ValueHead in the KataGo codebase.

    For historical reasons, I am mimicking the flow of the PolicyHead for now.
    """
    def __init__(self, name: str, c_in: int, c_hidden: int, shape: Shape):
        super(OwnershipHead, self).__init__(name, OwnershipTarget())

        self.shape = shape
        assert len(shape) == 3, f'Unexpected shape {shape}, Conv2d will not work'
        n_categories = shape[0]

        self.act = F.relu
        self.conv1 = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(c_hidden, n_categories, kernel_size=1, bias=False)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out


MODULE_MAP = {
    'ActionValueHead': ActionValueHead,
    'ConvBlock': ConvBlock,
    'ConvBlockWithGlobalPooling': ConvBlockWithGlobalPooling,
    'ResBlock': ResBlock,
    'ResBlockWithGlobalPooling': ResBlockWithGlobalPooling,
    'PolicyHead': PolicyHead,
    'ScoreHead': ScoreHead,
    'ValueHead': ValueHead,
    'OwnershipHead': OwnershipHead,
    }


@dataclass
class ModuleSpec:
    type: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    shape_info_dict: ShapeInfoDict
    stem: ModuleSpec
    blocks: List[ModuleSpec]
    heads: List[ModuleSpec]
    loss_weights: Dict[str, float]

    def validate(self):
        for spec in [self.stem] + self.blocks + self.heads:
            assert spec.type in MODULE_MAP, f'Unknown module type {spec.type}'


ModelConfigGenerator = Callable[[ShapeInfoDict], ModelConfig]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()

        config.validate()

        self.config = config
        self.stem = Model._construct_module(config.stem)
        self.blocks = nn.ModuleList(map(Model._construct_module, config.blocks))
        self.heads = nn.ModuleList(map(Model._construct_module, config.heads))
        self.loss_weights = config.loss_weights

        self.validate()

    @property
    def shape_info_dict(self) -> ShapeInfoDict:
        return self.config.shape_info_dict

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
        return tuple(head(out) for head in self.heads)

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
        assert self.heads[2].name == 'action-value', 'The third head must be the action-value head'

        for name in self.loss_weights:
            assert name in head_names, f'Loss weight for unknown head {name}'

        for name in head_names:
            assert name in self.loss_weights, f'Loss weight missing for head {name}'

        targets = [t for t in self.shape_info_dict.keys() if t != 'input']
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
        clone.heads = clone.heads[:3]

        clone.to('cpu')
        clone.eval()
        forward_shape = tuple([1] + list(self.shape_info_dict['input'].shape))
        example_input = torch.zeros(forward_shape)

        # Perform the actual trace/save in a separate process to avoid memory leak
        # See: https://github.com/pytorch/pytorch/issues/35600
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as pickle_file:
            pickle.dump((clone, example_input), pickle_file)
            pickle_file.close()

            script = os.path.join(Repo.root(), 'py/alphazero/scripts/jit_tracer.py')
            cmd = f'python {script} {pickle_file.name} {filename}'
            rc = os.system(cmd)
            assert rc == 0, f'Error saving model to {filename}'

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
