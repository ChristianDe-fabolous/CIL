import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder_transformer import TransformerDecoder
from .decoder_conv import ConvDecoder


class DepthModel(nn.Module):
    def __init__(
        self,
        decoder_type: str = "transformer",
        pretrained: bool = True,
        img_size: int = 560,
        patch_size: int = 16,
        decoder_blocks: int = 5,
        embed_dim: int = 384,
        num_heads: int = 6,
    ):
        super().__init__()
        self.encoder = ViTEncoder(pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        patch_grid = img_size // patch_size

        if decoder_type == "transformer":
            self.decoder = TransformerDecoder(
                embed_dim=embed_dim,
                num_blocks=decoder_blocks,
                num_heads=num_heads,
                patch_grid=patch_grid,
                patch_size=patch_size,
            )
        elif decoder_type == "conv":
            self.decoder = ConvDecoder(embed_dim=embed_dim, patch_grid=patch_grid, patch_size=patch_size)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
