import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_transformer import TransformerBlock


class DepthPatcher(nn.Module):
    """Avg-pool depth map into patches and project to embed_dim."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, patch_size)
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.pool(depth)                              # (B, 1, G, G)
        B, _, G, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, G * G, 1)  # (B, N, 1)
        return self.proj(x)                               # (B, N, D)


class CrossAttentionBlock(nn.Module):
    """Query tokens cross-attend to key-value tokens (context)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2   = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))[0]
        q = q + self.mlp(self.norm2(q))
        return q


class InContextDepthModel(nn.Module):
    """
    Predict depth for a query image given K context (image, depth) pairs.

    The shared ViT encoder processes both query and context images.
    Context depth maps are patch-projected and added to context image tokens.
    Cross-attention blocks let query tokens attend to context tokens.
    Self-attention blocks then refine the query representation before the depth head.

    Forward:
        query_img  : (B, 3, H, W)
        ctx_imgs   : (B, K, 3, H, W)
        ctx_depths : (B, K, 1, H, W)
    Returns:
        depth : (B, 1, H, W)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_cross_blocks: int = 3,
        num_self_blocks: int = 5,
        patch_grid: int = 35,
        patch_size: int = 16,
    ):
        super().__init__()
        self.encoder       = encoder
        self.depth_patcher = DepthPatcher(embed_dim, patch_size)
        self.cross_blocks  = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_cross_blocks)
        ])
        self.self_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_self_blocks)
        ])
        self.norm       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, 1)
        self.patch_grid = patch_grid
        self.patch_size = patch_size

    def _encode_context(
        self, ctx_imgs: torch.Tensor, ctx_depths: torch.Tensor
    ) -> torch.Tensor:
        B, K, C, H, W = ctx_imgs.shape
        img_tok = self.encoder(ctx_imgs.reshape(B * K, C, H, W))        # (B*K, N, D)
        dep_tok = self.depth_patcher(ctx_depths.reshape(B * K, 1, H, W)) # (B*K, N, D)
        N, D = img_tok.shape[1], img_tok.shape[2]
        return (img_tok + dep_tok).reshape(B, K * N, D)                  # (B, K*N, D)

    def forward(
        self,
        query_img: torch.Tensor,
        ctx_imgs: torch.Tensor,
        ctx_depths: torch.Tensor,
    ) -> torch.Tensor:
        q   = self.encoder(query_img)                       # (B, N, D)
        ctx = self._encode_context(ctx_imgs, ctx_depths)    # (B, K*N, D)

        for block in self.cross_blocks:
            q = block(q, ctx)
        for block in self.self_blocks:
            q = block(q)

        q     = self.norm(q)
        depth = self.head(q)                                # (B, N, 1)
        B     = depth.shape[0]
        depth = depth.reshape(B, self.patch_grid, self.patch_grid, 1).permute(0, 3, 1, 2)
        depth = F.interpolate(depth, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        return F.relu(depth) + 1e-3
