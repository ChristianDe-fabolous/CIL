import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder_transformer import TransformerDecoder
from .decoder_conv import ConvDecoder
from .upsampler import UpsamplerHead


class DepthModel(nn.Module):
    def __init__(
        self,
        decoder_type: str = "transformer",
        encoder_size: str = "small",
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 16,
        decoder_blocks: int = 5,
        output_size: int = None,  # None = no learned upsampler
    ):
        super().__init__()
        self.encoder = ViTEncoder(size=encoder_size, pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        embed_dim  = self.encoder.embed_dim
        num_heads  = self.encoder.num_heads
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

        self.upsampler = UpsamplerHead(output_size) if output_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        if self.upsampler is not None:
            out = self.upsampler(out)
        return out
