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
        self.patch_size: int = patch_size
        self.patch_grid: int = img_size // patch_size  # 35

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)  # (B, 1+N, D) with CLS prepended
        return tokens[:, 1:, :]                  # drop CLS → (B, N, D)


class DepthAnythingEncoder(nn.Module):
    """DINOv2 backbone from Depth Anything V2 (patch_size=14, embed_dim=768 for Base)."""

    _HF_NAMES = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, size: str = "base", img_size: int = 560):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        hf_name = self._HF_NAMES[size]
        da = AutoModelForDepthEstimation.from_pretrained(hf_name)
        self.backbone = da.backbone          # Dinov2Model
        self.patch_size: int = self.backbone.config.patch_size   # 14
        self.embed_dim: int  = self.backbone.config.hidden_size  # 384/768/1024
        self.patch_grid: int = img_size // self.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]  # drop CLS → (B, N, D)
