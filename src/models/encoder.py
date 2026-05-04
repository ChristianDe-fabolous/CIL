import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, img_size: int = 560, patch_size: int = 16):
        super().__init__()
        self.model = timm.create_model(
            f"vit_small_patch{patch_size}_224",
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )
        self.embed_dim: int = self.model.embed_dim   # 384
        self.patch_grid: int = img_size // patch_size  # 35

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)  # (B, 1+N, D) with CLS prepended
        return tokens[:, 1:, :]                  # drop CLS → (B, N, D)
