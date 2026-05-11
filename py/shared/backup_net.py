"""
BackupNet: the lightweight CPU-side neural network used by BetaZero to compute the per-node
posterior estimates Q(p) and W(p) from a per-node accumulator, the static latent z_s, and the
LoTE/LoTV baselines Qs*/Ws* (active-seat scalars; the trailing 's' stands for 'seat').

See docs/BetaZero.pdf, Section 4.3 for the full design.

Architecture (training-time forward):

    h1 = ReLU(W_1 @ [accumulator; z_s; Qs*; Ws*] + b_1)
    h2 = ReLU(W_2 @ h1 + b_2)
    [Q, W] = W_3 @ h2 + b_3

The per-child embedding step (e_i = ReLU(W_e @ [Q_i, W_i, N_i, P_i, z_a,i] + b_e), then
masked sum over children) lives separately in `ChildEmbeddingHead` and `AccumulatorHead`
(see net_modules.py).

BackupNet is declared as a regular DAG node in `ModelConfig` (no special-casing) with
parents `[accumulator, static_latent, Qs_star, Ws_star]`, where `Qs_star` and `Ws_star` are
declared in the ModelConfig's `external_inputs=[...]` list. It is *not* part of the
exported inference graph: it gets dropped by `ModelConfig.trim()` since no inference target
depends on it. Its weights still reach the C++ NNUE engine because `Model.save_model` walks
the un-trimmed model collecting `collect_graph_initializers(out)` contributions and embeds
them as orphan ONNX initializers prefixed `nnue/`.
"""
from typing import Dict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


# Number of context scalars passed to layer 1 alongside the accumulator and z_s: [Qs*, Ws*].
NUM_STATIC_SCALARS = 2


class BackupNet(nn.Module):
    """
    Output layout: (B, value_dim + 1) where:
      * columns [0 : value_dim] are Q logits in the same format as the base-NN's value head
        (e.g. WLD logits for WinLossDrawValueHead, WL logits for WinLossValueHead, ...);
      * column  [value_dim]      is the W (uncertainty) scalar.

    Predicting Q in the value head's native logit format lets BackupLossTerm reuse the value
    head's loss function (CrossEntropy) and `to_win_share` exactly, keeping calibration with
    the base-NN's value loss directly comparable. The C++ NNUE engine converts Q logits to a
    scalar win-share at consume-time via the same `to_win_share` semantics.
    """
    def __init__(
        self,
        value_dim: int,
        static_latent_dim: int,
        embed_dim: int,
        layer1_dim: int,
        layer2_dim: int,
    ):
        super().__init__()
        self.value_dim = value_dim
        self.static_latent_dim = static_latent_dim
        self.embed_dim = embed_dim
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.output_dim = value_dim + 1

        # Layer 1: accumulator + z_s + [Qs*, Ws*]
        layer1_in = embed_dim + static_latent_dim + NUM_STATIC_SCALARS
        self.layer1 = nn.Linear(layer1_in, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.out = nn.Linear(layer2_dim, self.output_dim)

    def forward(
        self,
        accumulator: torch.Tensor,   # (B, embed_dim)
        z_s: torch.Tensor,           # (B, d_s)
        Qs_star: torch.Tensor,       # (B,)       active-seat LoTE baseline
        Ws_star: torch.Tensor,       # (B,)       active-seat LoTV baseline
    ) -> torch.Tensor:
        """
        Returns (B, value_dim + 1): [Q logits ...; W scalar].
        """
        Qs = Qs_star.view(-1, 1).to(accumulator.dtype)
        Ws = Ws_star.view(-1, 1).to(accumulator.dtype)
        h0 = torch.cat([accumulator, z_s, Qs, Ws], dim=1)  # (B, embed + d_s + 2)
        h1 = F.relu(self.layer1(h0))
        h2 = F.relu(self.layer2(h1))
        return self.out(h2)                                # (B, value_dim + 1)

    def collect_graph_initializers(self, out: Dict[str, np.ndarray]) -> None:
        """
        Populates `out` with this module's parameters as orphan ONNX initializers (the caller
        prepends `nnue/`). The C++ NNUE engine consumes these by name via core::ReceivedModel.

        Layout (all float32):
            layer1.weight : (layer1_dim, embed_dim + d_s + 2)
            layer1.bias   : (layer1_dim,)
            layer2.weight : (layer2_dim, layer1_dim)
            layer2.bias   : (layer2_dim,)
            out.weight    : (value_dim + 1, layer2_dim)
            out.bias      : (value_dim + 1,)
        """
        for name, param in self.named_parameters():
            arr = param.detach().cpu().numpy().astype(np.float32, copy=False)
            out[name] = arr
