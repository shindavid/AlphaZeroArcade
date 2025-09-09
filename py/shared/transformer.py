import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Small utilities
# -----------------------

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
    """Dynamic, state-conditioned supplemental logits: (B,H,T,T)"""
    def __init__(self, Dm: int, H: int, T: int, compress_dim: int = 32, shared_dim: int = 256,
                 shared_layer: nn.Linear | None = None):
        super().__init__()
        self.T, self.H = T, H
        self.proj_token  = nn.Linear(Dm, compress_dim, bias=True)
        self.proj_global = nn.Linear(T * compress_dim, shared_dim, bias=True)
        self.per_head    = nn.Linear(shared_dim, H * shared_dim, bias=True)
        self.norm1 = nn.LayerNorm(shared_dim)
        self.norm2 = nn.LayerNorm(shared_dim)
        self.shared = shared_layer if shared_layer is not None else nn.Linear(shared_dim, T * T, bias=False)

    def forward(self, x):  # x: (B,T,Dm)
        B = x.size(0)
        z = self.proj_token(x)                     # (B,T,32)
        g = self.proj_global(z.reshape(B, -1))     # (B,256)
        g = F.silu(g)
        g = self.norm1(g)

        u = self.per_head(g)   # (B,H*256)
        u = F.silu(u)
        u = u.view(B, self.H, -1)   # (B,H,256)
        u = self.norm2(u)
        S = self.shared(u).view(B, self.H, self.T, self.T)  # (B,H,T,T)
        return S

class MultiheadAttentionWithExtras(nn.Module):
    """
    MHA with:
      - QKV no bias
      - optional Static Bias B (H,T,T)
      - optional Shaw pairwise (aQ,aK,aV) in logits/values
      - optional Smolgen supplemental logits S(x)
    """
    def __init__(self, Dm: int, H: int, T: int,
                 use_static_bias: bool = True,
                 use_shaw: bool = True,
                 use_smolgen: bool = True,
                 smolgen_shared: nn.Linear | None = None,
                 smolgen_shared_dim: int = 256,
                 smolgen_compress_dim: int = 32,
                 n_layers: int = 1):

        super().__init__()
        assert Dm % H == 0
        self.Dm, self.H, self.T = Dm, H, T
        self.dh = Dm // H
        self.scale = 1.0 / math.sqrt(self.dh)
        self.res_scale = residual_scale(n_layers)

        self.Wq = nn.Linear(Dm, Dm, bias=False)
        self.Wk = nn.Linear(Dm, Dm, bias=False)
        self.Wv = nn.Linear(Dm, Dm, bias=False)
        self.Wo = nn.Linear(Dm, Dm, bias=True)  # keep bias after attention

        self.use_static = use_static_bias
        self.use_shaw   = use_shaw
        self.use_smol   = use_smolgen

        if self.use_static:
            self.static_bias = nn.Parameter(torch.zeros(H, T, T))  # (H,T,T)

        if self.use_shaw:
            # Full pairwise (H,T,T,dh)
            self.aQ = nn.Parameter(torch.zeros(H, T, T, self.dh))
            self.aK = nn.Parameter(torch.zeros(H, T, T, self.dh))
            self.aV = nn.Parameter(torch.zeros(H, T, T, self.dh))
            for p in (self.aQ, self.aK, self.aV):
                nn.init.trunc_normal_(p, std=0.02)

        if self.use_smol:
            self.smolgen = Smolgen(Dm=Dm, H=H, T=T, shared_layer=smolgen_shared,
                                   compress_dim=smolgen_compress_dim, shared_dim=smolgen_shared_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: (B,T,Dm)
        B, T, Dm = x.shape
        H, dh = self.H, self.dh

        Q = self.Wq(x).view(B, T, H, dh).transpose(1, 2)  # (B,H,T,dh)
        K = self.Wk(x).view(B, T, H, dh).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, dh).transpose(1, 2)

        # Base logits
        logits = torch.einsum('bhtd,bhsd->bhts', Q, K) * self.scale  # (B,H,T,T)

        # Static bias
        if self.use_static:
            logits = logits + self.static_bias.unsqueeze(0)  # (B,H,T,T)

        # Shaw pairwise
        if self.use_shaw:
            logits = (logits
                      + torch.einsum('bhtd,htsd->bhts', Q, self.aK) * self.scale
                      + torch.einsum('htsd,bhsd->bhts', self.aQ, K) * self.scale
                      + torch.einsum('htsd,htsd->hts',  self.aQ, self.aK).unsqueeze(0) * self.scale)

        # Smolgen dynamic bias
        if self.use_smol:
            logits = logits + self.smolgen(x)  # (B,H,T,T)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhts,bhsd->bhtd', attn, V)  # (B,H,T,dh)
        if self.use_shaw:
            out = out + torch.einsum('bhts,htsd->bhtd', attn, self.aV)

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
    def __init__(self, Dm: int, H: int, T: int, Dff: int, n_layers: int,
                 use_static_bias=True, use_shaw=True, use_smolgen=True, smolgen_shared=None,
                 smolgen_shared_dim: int = 256, smolgen_compress_dim: int = 32):
        super().__init__()
        self.mha = MultiheadAttentionWithExtras(Dm, H, T,
                                                use_static_bias=use_static_bias,
                                                use_shaw=use_shaw,
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


class ChessformerBlock(nn.Module):
    def __init__(self, input_shape, embed_dim: int, n_heads: int, n_layers: int,
                 n_output_channels: int,
                 use_static_bias: bool = True,
                 use_shaw: bool = True,
                 use_smolgen: bool = True,
                 smolgen_compress_dim: int = 32,
                 smolgen_shared_dim: int = 256,
                 ffn_multiplier: float = 1.0):
        """
        input_shape: (B, C, H, W)
        """
        super(ChessformerBlock, self).__init__()

        H, W = input_shape[1], input_shape[2]
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
                EncoderLayer(Dm=embed_dim, H=Hh, T=self.board_size, Dff=Dff, n_layers=n_layers,
                             use_static_bias=use_static_bias,
                             use_shaw=use_shaw,
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
        assert H == self.H and W == self.W, f"Expected HxW = {self.H}x{self.W}, got {H}x{W}"

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, T, C)

        x = self.input_embed(x)  # (B, T, Dm)
        x = self.embed_act(x)
        x = self.gating(x)  # (B, T, Dm)

        for layer in self.transformer_encoder:
            x = layer(x)  # (B, T, Dm)

        x = self.output_projection(x)  # (B, T, n_output_channels)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)

        return x
