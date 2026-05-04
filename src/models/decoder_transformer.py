import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_blocks: int = 5,
        num_heads: int = 6,
        patch_grid: int = 35,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, D) → depth: (B, 1, H, W)"""
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        depth = self.head(tokens)                                              # (B, N, 1)
        B = depth.shape[0]
        depth = depth.reshape(B, self.patch_grid, self.patch_grid, 1).permute(0, 3, 1, 2)
        depth = F.interpolate(depth, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        return F.relu(depth) + 1e-3
