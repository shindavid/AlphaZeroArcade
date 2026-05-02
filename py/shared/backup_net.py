"""
BackupNet: the lightweight CPU-side neural network used by BetaZero to compute the per-node
posterior estimates Q(p) and W(p) from a per-node accumulator, the static latent z_s, and the
LoTE/LoTV baselines Q*/W*.

See docs/BetaZero.pdf, Section 4.3 for the full design.

Architecture (training-time forward):

    h1 = ReLU(W_1 @ [accumulator; z_s; Q*; W*] + b_1)
    h2 = ReLU(W_2 @ h1 + b_2)
    [Q, W] = W_3 @ h2 + b_3

The per-child embedding step (e_i = ReLU(W_e @ [Q_i, W_i, N_i, P_i, z_a,i] + b_e), then
masked sum over children) lives separately in `ChildEmbeddingHead` and `AccumulatorHead`
(see net_modules.py).

BackupNet is declared as a regular DAG node in `ModelConfig` (no special-casing) with
parents `[accumulator, static_latent, input_Q_star, input_W_star]`. It is *not* part of the
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


# Output dim: [Q, W].  Constrained to scalar Q/W (1-player or zero-sum 2-player only).
OUTPUT_DIM = 2

# Number of context scalars passed to layer 1 alongside the accumulator and z_s: [Q*, W*].
NUM_STATIC_SCALARS = 2


class BackupNet(nn.Module):
    def __init__(
        self,
        static_latent_dim: int,
        embed_dim: int,
        layer1_dim: int,
        layer2_dim: int,
    ):
        super().__init__()
        self.static_latent_dim = static_latent_dim
        self.embed_dim = embed_dim
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim

        # Layer 1: accumulator + z_s + [Q*, W*]
        layer1_in = embed_dim + static_latent_dim + NUM_STATIC_SCALARS
        self.layer1 = nn.Linear(layer1_in, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.out = nn.Linear(layer2_dim, OUTPUT_DIM)

    def forward(
        self,
        accumulator: torch.Tensor,   # (B, embed_dim)
        z_s: torch.Tensor,           # (B, d_s)
        Q_star: torch.Tensor,        # (B,)       LoTE baseline
        W_star: torch.Tensor,        # (B,)       LoTV baseline
    ) -> torch.Tensor:
        """
        Returns (B, 2) tensor: [Q, W] columns.
        """
        Qs = Q_star.view(-1, 1).to(accumulator.dtype)
        Ws = W_star.view(-1, 1).to(accumulator.dtype)
        h0 = torch.cat([accumulator, z_s, Qs, Ws], dim=1)  # (B, embed + d_s + 2)
        h1 = F.relu(self.layer1(h0))
        h2 = F.relu(self.layer2(h1))
        return self.out(h2)                                # (B, 2)

    def collect_graph_initializers(self, out: Dict[str, np.ndarray]) -> None:
        """
        Populates `out` with this module's parameters as orphan ONNX initializers (the caller
        prepends `nnue/`). The C++ NNUE engine consumes these by name via core::ReceivedModel.

        Layout (all float32):
            layer1.weight : (layer1_dim, embed_dim + d_s + 2)
            layer1.bias   : (layer1_dim,)
            layer2.weight : (layer2_dim, layer1_dim)
            layer2.bias   : (layer2_dim,)
            out.weight    : (2, layer2_dim)
            out.bias      : (2,)
        """
        for name, param in self.named_parameters():
            arr = param.detach().cpu().numpy().astype(np.float32, copy=False)
            out[name] = arr
