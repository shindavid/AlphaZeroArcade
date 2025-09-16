import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def residual_scale(n_layers: int) -> float:
    return 1.0 / math.sqrt(2.0 * max(1, n_layers))


class MAGating(nn.Module):
    def __init__(self, T: int, D: int):
        super().__init__()
        self.pos_add = nn.Parameter(torch.zeros(T, D))
        self.pos_mul = nn.Parameter(torch.zeros(T, D))
        nn.init.trunc_normal_(self.pos_add, std=0.02)
        nn.init.trunc_normal_(self.pos_mul, std=0.02)

    def forward(self, U):
        mult = F.softplus(1.0 + self.pos_mul)
        return U * mult + self.pos_add


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):  # x: (..., D)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * x * norm


class Smolgen(nn.Module):
    """Dynamic, state-conditioned supplemental logits: (B,Hh,T,T)"""
    def __init__(self, Dm: int, Hh: int, T: int, compress_dim: int = 32, shared_dim: int = 256,
                 shared_layer: nn.Linear | None = None):
        super().__init__()
        self.T, self.Hh = T, Hh
        self.proj_token  = nn.Linear(Dm, compress_dim, bias=True)
        self.proj_global = nn.Linear(T * compress_dim, shared_dim, bias=True)
        self.per_head    = nn.Linear(shared_dim, Hh * shared_dim, bias=True)
        self.norm1 = nn.LayerNorm(shared_dim)
        self.norm2 = nn.LayerNorm(shared_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.shared = shared_layer if shared_layer is not None else nn.Linear(shared_dim, T * T, bias=False)

    def forward(self, x):  # x: (B,T,Dm)
        B = x.size(0)
        z = self.proj_token(x)                     # (B,T,32)
        g = self.proj_global(z.reshape(B, -1))     # (B,shared_dim)
        g = F.silu(g)
        g = self.norm1(g)

        u = self.per_head(g)  # (B,Hh*shared_dim)
        u = F.silu(u)
        u = u.view(B, self.Hh, -1)  # (B,Hh,shared_dim)
        u = self.norm2(u)
        S = self.shared(u).view(B, self.Hh, self.T, self.T)  # (B,Hh,T,T)
        return self.gate * S


class MultiheadAttentionWithExtras(nn.Module):
    """
    MHA with:
      - QKV no bias
      - optional Static Bias B (H,T,T)
      - optional Shaw pairwise (aQ,aK,aV) in logits/values
      - optional Smolgen supplemental logits S(x)
    """
    def __init__(self, Dm: int, Hh: int, T: int,
                 use_static_bias: bool = True,
                 use_shaw: bool = True,
                 use_rpe: bool = False,
                 rpe_factorizer: Optional[torch.Tensor] = None,
                 use_smolgen: bool = True,
                 smolgen_shared: Optional[nn.Linear] = None,
                 smolgen_shared_dim: int = 256,
                 smolgen_compress_dim: int = 32,
                 n_layers: int = 1):

        super().__init__()

        self.Dm, self.Hh, self.T = Dm, Hh, T
        # assert int(Dm % Hh) == 0, f"Dm ({Dm}) must be divisible by Hh ({Hh})"

        self.dh = Dm // Hh
        self.scale = 1.0 / math.sqrt(self.dh)
        self.res_scale = residual_scale(n_layers)

        self.Wq = nn.Linear(Dm, Dm, bias=False)
        self.Wk = nn.Linear(Dm, Dm, bias=False)
        self.Wv = nn.Linear(Dm, Dm, bias=False)
        self.Wo = nn.Linear(Dm, Dm, bias=True)  # keep bias after attention

        self.use_static = use_static_bias
        self.use_shaw   = use_shaw
        self.use_rpe    = use_rpe
        self.use_smol   = use_smolgen

        if self.use_static:
            self.static_bias = nn.Parameter(torch.zeros(Hh, T, T))  # (Hh,T,T)

        if self.use_shaw:
            # Full pairwise (Hh,T,T,dh)
            self.aQ = nn.Parameter(torch.zeros(Hh, T, T, self.dh))
            self.aK = nn.Parameter(torch.zeros(Hh, T, T, self.dh))
            self.aV = nn.Parameter(torch.zeros(Hh, T, T, self.dh))
            for p in (self.aQ, self.aK, self.aV):
                nn.init.trunc_normal_(p, std=0.02)

        if self.use_rpe:
            assert rpe_factorizer is not None, "rpe_factorizer must be provided if use_rpe is True"
            self.register_buffer('rpe_factorizer', rpe_factorizer)  # (num_bins, T*T)

            num_bins = rpe_factorizer.size(0)
            Hd = Hh * self.dh

            # already have:
            self.rpe_q = nn.Parameter(torch.zeros(Hd, num_bins))
            self.rpe_k = nn.Parameter(torch.zeros(Hd, num_bins))
            self.rpe_v = nn.Parameter(torch.zeros(Hd, num_bins))
            nn.init.trunc_normal_(self.rpe_q, std=0.02)
            nn.init.trunc_normal_(self.rpe_k, std=0.02)
            nn.init.trunc_normal_(self.rpe_v, std=0.02)


        if self.use_smol:
            self.smolgen = Smolgen(Dm=Dm, Hh=Hh, T=T, shared_layer=smolgen_shared,
                                   compress_dim=smolgen_compress_dim, shared_dim=smolgen_shared_dim)

        self.dropout = nn.Dropout(0.1)

    def _expand_rpe_logits(self, W_small: torch.Tensor) -> torch.Tensor:
        # (Hh*dh, T*T)
        rpe_flat = torch.matmul(W_small, self.rpe_factorizer)
        return rpe_flat.view(self.Hh, self.dh, self.T, self.T).permute(0, 2, 3, 1)  # (Hh, T, T, dh)

    def forward(self, x):  # x: (B,T,Dm)
        B, T, Dm = x.shape
        Hh, dh = self.Hh, self.dh

        Q = self.Wq(x).view(B, T, Hh, dh).transpose(1, 2)  # (B,Hh,T,dh)
        K = self.Wk(x).view(B, T, Hh, dh).transpose(1, 2)  # (B,Hh,T,dh)
        V = self.Wv(x).view(B, T, Hh, dh).transpose(1, 2)  # (B,Hh,T,dh)

        # Base logits
        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,Hh,T,T)

        # Static bias
        if self.use_static:
            logits = logits + self.static_bias.unsqueeze(0)  # (B,Hh,T,T)

        # Shaw pairwise
        if self.use_shaw:
            logits = (logits
                      + torch.einsum('bhtd,htsd->bhts', Q, self.aK) * self.scale
                      + torch.einsum('htsd,bhsd->bhts', self.aQ, K) * self.scale
                      + torch.einsum('htsd,htsd->hts',  self.aQ, self.aK).unsqueeze(0) * self.scale)

        # RPE
        if self.use_rpe:
            rpe_q = self._expand_rpe_logits(self.rpe_q)  # (Hh, T, T, dh)
            rpe_k = self._expand_rpe_logits(self.rpe_k)  # (Hh, T, T, dh)
            logits = (logits
                      + torch.einsum('bhtd,htsd->bhts', Q, rpe_k) * self.scale
                      + torch.einsum('htsd,bhsd->bhts', rpe_q, K) * self.scale)

        # Smolgen dynamic bias
        if self.use_smol:
            logits = logits + self.smolgen(x)  # (B,Hh,T,T)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B,Hh,T,dh)
        if self.use_shaw:
            out = out + torch.einsum('bhts,htsd->bhtd', attn, self.aV)

        if self.use_rpe:
            rpe_v = self._expand_rpe_logits(self.rpe_v)  # (Hh, dh, T, T)
            out = out + torch.einsum('bhts,htsd->bhtd', attn, rpe_v)

        out = out.transpose(1, 2).contiguous().view(B, T, Dm)  # (B,T,Dm)
        out = self.dropout(self.Wo(out)) * self.res_scale
        return out


class FFN(nn.Module):
    def __init__(self, Dm: int, Dff: int, n_layers: int):
        super().__init__()
        self.fc1 = nn.Linear(Dm, Dff, bias=True)
        self.act = Mish()
        self.fc2 = nn.Linear(Dff, Dm, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.res_scale = residual_scale(n_layers)

    def forward(self, x):
        y = self.fc2(self.act(self.fc1(x)))
        return self.dropout(y) * self.res_scale


class EncoderLayer(nn.Module):
    def __init__(self, Dm: int, Hh: int, T: int, Dff: int, n_layers: int,
                 use_static_bias=True, use_shaw=True,
                 use_rpe: bool = False, rpe_factorizer: Optional[torch.Tensor] = None,
                 use_smolgen=True, smolgen_shared=None,
                 smolgen_shared_dim: int = 256, smolgen_compress_dim: int = 32):
        super().__init__()
        self.mha = MultiheadAttentionWithExtras(Dm, Hh, T,
                                                use_static_bias=use_static_bias,
                                                use_shaw=use_shaw,
                                                use_rpe=use_rpe,
                                                rpe_factorizer=rpe_factorizer,
                                                use_smolgen=use_smolgen,
                                                smolgen_shared=smolgen_shared,
                                                smolgen_shared_dim=smolgen_shared_dim,
                                                smolgen_compress_dim=smolgen_compress_dim,
                                                n_layers=n_layers)
        self.ffn = FFN(Dm, Dff, n_layers)
        self.norm1 = RMSNorm(Dm)
        self.norm2 = RMSNorm(Dm)

    def forward(self, x):  # x: (B,T,Dm)
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


def build_rpe_factorizer_2d(H, W) -> torch.Tensor: # ((2*H-1)*(2*W-1) , (H*W)*(H*W))
    """
    Build a binary factorizer matrix M for 2D relative position encoding on an H×W grid.

    Shape:
        M ∈ {0,1}^{(2H-1)*(2W-1) , (H*W)*(H*W)}

    Meaning:
        • Rows correspond to all possible relative offsets (dy, dx) between two squares:
              dy = i - k,   dx = j - l
          where (i, j) is the query square (row i, column j)
          and (k, l) is the key square (row k, column l).
          dy ∈ [-(H-1), ..., H-1],  dx ∈ [-(W-1), ..., W-1].
          Total number of rows = (2H-1) * (2W-1).

        • Columns correspond to all ordered query–key square pairs ((i, j), (k, l))
          on the H×W grid. There are (H*W)² such pairs.

    Indexing:
        • Let R_h = H - 1 and R_w = W - 1.
        • Row index = (dy + R_h) * (2W - 1) + (dx + R_w).
        • Query index: q_idx = i * W + j.
        • Key index:   k_idx = k * W + l.
        • Column index = q_idx * (H*W) + k_idx.

    Entry:
        M[row, col] = 1 means the query–key pair ((i, j), (k, l))
                      has relative displacement (dy, dx).
        Each column has exactly one '1', selecting the unique relative-offset bin
        for that (query, key) pair.
    """
    assert H > 0 and W > 0, "H and W must be positive integers."

    N = H * W
    Rh, Rw = H - 1, W - 1
    n_rows = (2 * H - 1) * (2 * W - 1)
    n_cols = N * N

    # Query and key coordinates
    ys = torch.arange(H)
    xs = torch.arange(W)
    QY, QX = torch.meshgrid(ys, xs, indexing="ij")
    KY, KX = QY, QX

    QY = QY.reshape(-1)  # (N,)
    QX = QX.reshape(-1)
    KY = KY.reshape(-1)
    KX = KX.reshape(-1)

    # Relative displacements
    dy = QY.unsqueeze(1) - KY.unsqueeze(0)  # (N, N)
    dx = QX.unsqueeze(1) - KX.unsqueeze(0)  # (N, N)

    # Row indices (for relative bins)
    row_idx = (dy + Rh) * (2 * W - 1) + (dx + Rw)
    row_idx = row_idx.reshape(-1).long()  # (N*N,)

    # Column indices (for (q,k) pairs)
    q_idx = torch.arange(N).unsqueeze(1).expand(N, N)
    k_idx = torch.arange(N).unsqueeze(0).expand(N, N)
    col_idx = (q_idx * N + k_idx).reshape(-1).long()  # (N*N,)

    # Construct binary factorizer
    M = torch.zeros(n_rows, n_cols)
    M.index_put_((row_idx, col_idx), torch.ones_like(row_idx, dtype=M.dtype), accumulate=False)

    return M


class ChessformerBlock(nn.Module):
    def __init__(self, input_shape, embed_dim: int, n_heads: int, n_layers: int,
                 n_output_channels: int,
                 use_static_bias: bool = True,
                 use_shaw: bool = True,
                 use_rpe: bool = False,
                 use_smolgen: bool = True,
                 smolgen_compress_dim: int = 32,
                 smolgen_shared_dim: int = 256,
                 ffn_multiplier: float = 1.0):
        """
        input_shape: (C, H, W)
        """
        super(ChessformerBlock, self).__init__()

        H, W = input_shape[1], input_shape[2]
        rpe_factorizer = build_rpe_factorizer_2d(H, W)

        self.H, self.W = H, W
        self.board_size = H * W  # T
        n_input_channels = input_shape[0]
        Hh = n_heads
        Dff = int(ffn_multiplier * embed_dim)

        self.input_embed = nn.Linear(n_input_channels, embed_dim, bias=True)
        self.embed_act = Mish()
        self.gating = MAGating(self.board_size, embed_dim)

        self.smolgen_shared = nn.Linear(smolgen_shared_dim, self.board_size * self.board_size, bias=False) if use_smolgen else None

        layers = []
        for _ in range(n_layers):
            layers.append(
                EncoderLayer(Dm=embed_dim, Hh=Hh, T=self.board_size, Dff=Dff, n_layers=n_layers,
                             use_static_bias=use_static_bias,
                             use_shaw=use_shaw,
                             use_rpe=use_rpe,
                             rpe_factorizer=rpe_factorizer,
                             use_smolgen=use_smolgen,
                             smolgen_compress_dim=smolgen_compress_dim,
                             smolgen_shared_dim=smolgen_shared_dim,
                             smolgen_shared=self.smolgen_shared)
            )
        self.transformer_encoder = nn.ModuleList(layers)
        self.output_projection = nn.Linear(embed_dim, n_output_channels, bias=True)

    def forward(self, x):
        """
          (B, C, H, W) -> (B, T, C) -> ... -> (B, T, n_out) -> (B, n_out, H, W)
        """
        (B, C, H, W) = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, T, C)

        x = self.input_embed(x)  # (B, T, Dm)
        x = self.embed_act(x)
        x = self.gating(x)  # (B, T, Dm)

        for layer in self.transformer_encoder:
            x = layer(x)  # (B, T, Dm)

        x = self.output_projection(x)  # (B, T, n_output_channels)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)

        return x
