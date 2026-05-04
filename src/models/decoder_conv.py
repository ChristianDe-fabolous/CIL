import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ConvDecoder(nn.Module):
    """
    Progressive 2x upsample decoder.
    35x35 → 70 → 140 → 280 → 560
    """
    def __init__(self, embed_dim: int = 384, patch_grid: int = 35, patch_size: int = 16):
        super().__init__()
        assert patch_size == 16, "ConvDecoder assumes patch_size=16 (4 × 2x upsample stages)"
        self.patch_grid = patch_grid
        self.stages = nn.ModuleList([
            ConvBlock(384, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
        ])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, D) → depth: (B, 1, H, W)"""
        B, N, D = tokens.shape
        x = tokens.reshape(B, self.patch_grid, self.patch_grid, D).permute(0, 3, 1, 2)
        for stage in self.stages:
            x = stage(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.head(x)) + 1e-3
