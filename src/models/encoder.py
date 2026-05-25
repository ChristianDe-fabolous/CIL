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


class DA2Encoder(nn.Module):
    """
    Depth Anything V2 with DPT-style multi-layer feature extraction.

    Instead of taking only the last transformer layer (like DepthAnythingEncoder),
    this samples 4 evenly-spaced layers and averages them.  Early layers carry
    fine-grained spatial detail; late layers carry semantic depth context.
    Averaging gives the DiT denoiser access to both simultaneously.
    """

    _HF_NAMES = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, size: str = "base", img_size: int = 560):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        da = AutoModelForDepthEstimation.from_pretrained(self._HF_NAMES[size])
        self.backbone   = da.backbone
        self.patch_size: int = self.backbone.config.patch_size
        self.embed_dim:  int = self.backbone.config.hidden_size
        self.patch_grid: int = img_size // self.patch_size
        self.num_layers: int = self.backbone.config.num_hidden_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        # hidden_states[0] = patch embedding, [1..L] = transformer layers
        layers = out.hidden_states[1:]          # L tensors of (B, N+1, D)
        n      = len(layers)
        # 4 evenly-spaced indices: first, 1/3, 2/3, last
        idx    = [0, n // 3, 2 * n // 3, n - 1]
        selected = [layers[i][:, 1:, :] for i in idx]   # drop CLS each
        return torch.stack(selected, dim=0).mean(0)       # (B, N, D)
