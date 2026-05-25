import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_1d(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    args = x[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_1d(t, self.dim))


class ResBlock(nn.Module):
    """Conv ResBlock with AdaGN timestep conditioning (scale + shift)."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        groups = lambda ch: min(8, ch)
        self.norm1  = nn.GroupNorm(groups(in_ch),  in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)
        self.norm2  = nn.GroupNorm(groups(out_ch), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.conv2(F.silu(h)) + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """
    Spatial cross-attention: flatten (B, C, H, W) → (B, H*W, C),
    attend to DA2 context tokens, reshape back.
    """

    def __init__(self, ch: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(
            ch, num_heads, kdim=context_dim, vdim=context_dim, batch_first=True
        )
        self.proj = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)          # (B, H*W, C)
        h, _ = self.attn(h, context, context)
        h = self.proj(h).transpose(1, 2).reshape(B, C, H, W)
        return x + h


class DepthUNet(nn.Module):
    """
    Full-resolution U-Net denoiser for 560×560 depth maps.

    Encoder progressively downsamples 560→280→140→70→35 via avg-pool.
    Cross-attention to DA2 multi-layer features happens at the 35×35
    bottleneck where attending to 1225 spatial positions vs 1600 DA2
    tokens is cheap.
    Decoder upsamples back to 560×560 with skip connections.

    Operates in log-depth space — input/output are log(depth).
    """

    def __init__(
        self,
        context_dim: int = 768,
        base_ch:     int = 32,
        t_dim:       int = 256,
    ):
        super().__init__()
        c = base_ch
        self.t_embed = TimestepEmbedding(t_dim)

        # Encoder
        self.enc1 = ResBlock(1,    c,    t_dim)   # (B, c,   560, 560)
        self.enc2 = ResBlock(c,    c*2,  t_dim)   # (B, c*2, 280, 280)
        self.enc3 = ResBlock(c*2,  c*4,  t_dim)   # (B, c*4, 140, 140)
        self.enc4 = ResBlock(c*4,  c*8,  t_dim)   # (B, c*8,  70,  70)

        # Bottleneck at 35×35
        self.bot1  = ResBlock(c*8, c*8, t_dim)
        self.bot_ca = CrossAttentionBlock(c*8, context_dim, num_heads=4)
        self.bot2  = ResBlock(c*8, c*8, t_dim)

        # Decoder (skip channels added)
        self.dec4 = ResBlock(c*8 + c*8, c*4, t_dim)
        self.dec3 = ResBlock(c*4 + c*4, c*2, t_dim)
        self.dec2 = ResBlock(c*2 + c*2, c,   t_dim)
        self.dec1 = ResBlock(c   + c,   c,   t_dim)

        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out_conv = nn.Conv2d(c, 1, 1)

    def _resize_context(self, context: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reshape DA2 token grid to spatial, interpolate to (h, w), flatten back."""
        B, N, D = context.shape
        g = int(N ** 0.5)
        ctx = context.transpose(1, 2).reshape(B, D, g, g)
        ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)
        return ctx.flatten(2).transpose(1, 2)                 # (B, h*w, D)

    def forward(
        self,
        x:       torch.Tensor,   # (B, 1, H, W) noisy log-depth
        context: torch.Tensor,   # (B, N, context_dim) DA2 features
        t:       torch.Tensor,   # (B,) timestep indices
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)

        # Encode with 2× downsamples between stages
        h1 = self.enc1(x,                     t_emb)   # 560
        h2 = self.enc2(F.avg_pool2d(h1, 2),   t_emb)   # 280
        h3 = self.enc3(F.avg_pool2d(h2, 2),   t_emb)   # 140
        h4 = self.enc4(F.avg_pool2d(h3, 2),   t_emb)   # 70

        # Bottleneck
        b = F.avg_pool2d(h4, 2)                          # 35
        b = self.bot1(b, t_emb)
        b = self.bot_ca(b, self._resize_context(context, b.shape[2], b.shape[3]))
        b = self.bot2(b, t_emb)

        # Decode with nearest-neighbor upsample + skip concat
        up = lambda feat, skip: F.interpolate(feat, skip.shape[2:], mode="nearest")
        d4 = self.dec4(torch.cat([up(b,  h4), h4], 1), t_emb)
        d3 = self.dec3(torch.cat([up(d4, h3), h3], 1), t_emb)
        d2 = self.dec2(torch.cat([up(d3, h2), h2], 1), t_emb)
        d1 = self.dec1(torch.cat([up(d2, h1), h1], 1), t_emb)

        return self.out_conv(F.silu(self.out_norm(d1)))   # (B, 1, H, W)
