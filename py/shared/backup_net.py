"""
BackupNet: the lightweight CPU-side neural network used by BetaZero to compute the per-node
posterior estimates Q(p) and W(p) from a per-node accumulator, the static latent z_s, and the
LoTE/LoTV baselines Ss*/Ws* (Ss* is the active-seat-rotated WLD/WL distribution; Ws* is the
active-seat uncertainty scalar; the trailing 's' stands for 'seat').

See docs/BetaZero.pdf, Section 4.3 for the full design.

Architecture (training-time forward):

    h1     = ReLU(W_1 @ [accumulator; z_s; Ss*; Ws*] + b_1)
    h2     = ReLU(W_2 @ h1 + b_2)
    [dQ, dW] = W_3 @ h2 + b_3                              # "residual" MLP head

    Q_logits = dQ + log(clamp(Ss*))                        # value-head-shaped logits
    W        = dW + Ws*                                    # uncertainty scalar

The per-child embedding step (e_i = ReLU(W_e @ [Q_i, W_i, N_i, P_i, z_a,i] + b_e), then
masked sum over children) lives separately in `ChildEmbeddingHead` and `AccumulatorHead`
(see net_modules.py).

S/W-skip (BetaZero "AlphaZero passthrough"):
  The third linear layer (`self.out`) is zero-initialized so that, at the start of training,
  dQ = 0 and dW = 0. The output is then exactly (log(clamp(Ss*)), Ws*); softmax recovers
  Ss* (to within the clamp epsilon), and W = Ws* exactly. This makes the BackupNet's
  initial behavior identical to AlphaZero's children-average aggregation, so the
  bootstrap-anchor regime in BackupLossTerm has near-zero loss from step 1 instead of
  having to learn the passthrough from scratch (which proved very slow when Ss*/Ws* have
  low input variance, e.g. when self-play uses the uniform NN service in early
  generations). Training only needs to fit the residual.

BackupNet is declared as a regular DAG node in `ModelConfig` (no special-casing) with
parents `[accumulator, static_latent, Ss_star, Ws_star]`, where `Ss_star` and `Ws_star` are
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


# S/W-skip constants. These MUST match the C++ side (cpp/inline/beta0/BackupNNEvaluator.inl)
# byte-for-byte; the equivalence unit test verifies this.
#
#   SSTAR_CLAMP_EPS   - clamp Ss* into [eps, +inf) before taking log() to avoid -inf.
SSTAR_CLAMP_EPS = 1e-4


def sstar_to_logit_skip(Ss_star: torch.Tensor) -> torch.Tensor:
    """
    Return the (B, value_dim) logit-space skip such that, *after softmax*, the resulting
    distribution equals Ss* (to within the SSTAR_CLAMP_EPS cutoff).

    Ss_star is expected to be a non-negative distribution (rows sum to ~1) of shape
    (B, value_dim) in the active-seat-rotated frame.
    """
    return torch.log(Ss_star.clamp(min=SSTAR_CLAMP_EPS))


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

    See module docstring for the S/W-skip / zero-init scheme that makes this an architectural
    "AlphaZero passthrough" at initialization.
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
        assert value_dim >= 2, f'BackupNet requires value_dim >= 2, got {value_dim}'
        self.value_dim = value_dim
        self.static_latent_dim = static_latent_dim
        self.embed_dim = embed_dim
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.output_dim = value_dim + 1

        # Layer 1: accumulator + z_s + Ss* (value_dim) + Ws* (1)
        layer1_in = embed_dim + static_latent_dim + value_dim + 1
        self.layer1 = nn.Linear(layer1_in, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.out = nn.Linear(layer2_dim, self.output_dim)

        # Zero-init the residual head so initial output = (log(clamp(Ss*)), Ws*) exactly.
        # See module docstring ("S/W-skip").
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        accumulator: torch.Tensor,   # (B, embed_dim)
        z_s: torch.Tensor,           # (B, d_s)
        Ss_star: torch.Tensor,       # (B, value_dim)  active-seat-rotated LoTE baseline
        Ws_star: torch.Tensor,       # (B,)            active-seat LoTV baseline
    ) -> torch.Tensor:
        """
        Returns (B, value_dim + 1): [Q logits ...; W scalar].

        At init the MLP head is zero, so the returned values are exactly
        (log(clamp(Ss*)), Ws*) -- see module docstring.
        """
        Ss = Ss_star.view(-1, self.value_dim).to(accumulator.dtype)
        Ws = Ws_star.view(-1, 1).to(accumulator.dtype)
        h0 = torch.cat([accumulator, z_s, Ss, Ws], dim=1)  # (B, embed + d_s + value_dim + 1)
        h1 = F.relu(self.layer1(h0))
        h2 = F.relu(self.layer2(h1))
        mlp_out = self.out(h2)                              # (B, value_dim + 1)  -- residual

        # S/W-skip: add the log-clamp skip + Ws* to recover
        # (Q_logits, W) = (log(clamp(Ss*)), Ws*) when the residual is zero.
        q_skip = sstar_to_logit_skip(Ss).to(mlp_out.dtype)
        q_logits = mlp_out[:, :self.value_dim] + q_skip
        w_pred = mlp_out[:, self.value_dim] + Ws.view(-1)
        return torch.cat([q_logits, w_pred.unsqueeze(-1)], dim=-1)

    def collect_graph_initializers(self, out: Dict[str, np.ndarray]) -> None:
        """
        Populates `out` with this module's parameters as orphan ONNX initializers (the caller
        prepends `nnue/`). The C++ NNUE engine consumes these by name via core::ModelBundle.

        Layout (all float32):
            layer1.weight : (layer1_dim, embed_dim + d_s + value_dim + 1)
            layer1.bias   : (layer1_dim,)
            layer2.weight : (layer2_dim, layer1_dim)
            layer2.bias   : (layer2_dim,)
            out.weight    : (value_dim + 1, layer2_dim)
            out.bias      : (value_dim + 1,)

        Note that the S/W-skip is part of the architecture, not a learned weight; the C++
        side applies it identically (see cpp/inline/beta0/BackupNNEvaluator.inl).
        """
        for name, param in self.named_parameters():
            arr = param.detach().cpu().numpy().astype(np.float32, copy=False)
            out[name] = arr
