import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplerHead(nn.Module):
    """Bilinear upsample to output_size, then refine with two conv layers."""

    def __init__(self, output_size: int = 560):
        super().__init__()
        self.output_size = output_size
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return self.refine(x)
