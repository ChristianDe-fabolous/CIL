import torch
import torch.nn as nn
import timm


_VIT_CONFIGS = {
    "small": {"model": "vit_small_patch{patch_size}_224", "num_heads": 6},
    "base":  {"model": "vit_base_patch{patch_size}_224",  "num_heads": 12},
}


class ViTEncoder(nn.Module):
    def __init__(self, size: str = "small", pretrained: bool = True, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        assert size in _VIT_CONFIGS, f"Unknown ViT size {size!r}, choose from {list(_VIT_CONFIGS)}"
        model_name = _VIT_CONFIGS[size]["model"].format(patch_size=patch_size)
        self.model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, num_classes=0)
        self.embed_dim: int  = self.model.embed_dim
        self.num_heads: int  = _VIT_CONFIGS[size]["num_heads"]
        self.patch_size: int = patch_size
        self.patch_grid: int = img_size // patch_size

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
